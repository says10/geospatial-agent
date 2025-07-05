# src/geospatial_agent/infrastructure/tools/gdal_tools.py

from pathlib import Path
from typing import Dict, Any, Union, Optional
from loguru import logger

# Import Pydantic schemas for data representation and operation results
from geospatial_agent.domain.schemas import (
    RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult
)
# Import DataLoader for data acquisition and processing orchestration
from geospatial_agent.data_ingestion.data_loader import DataLoader
# Import Ontology models for tool parameters
from geospatial_agent.ontology.ontology import (
    ReprojectRasterParams, ConvertVectorFormatParams, ClipRasterParams,
    CalculatePixelStatisticsParams, ReadVectorInfoParams,
    DownloadBhoonidhiDataParams, DownloadOSMDataParams, ReadRasterInfoParams, # Added ReadRasterInfoParams
    SpatialReferenceSystem, ResamplingMethod, VectorFormat, GeospatialDataType
)

# Import GDAL operations from gis_backends
from geospatial_agent.infrastructure.gis_backends.gdal_ops.reproject_raster import reproject_raster
from geospatial_agent.infrastructure.gis_backends.gdal_ops.convert_vector_format import convert_vector_format
from geospatial_agent.infrastructure.gis_backends.gdal_ops.clip_raster import clip_raster_with_vector
from geospatial_agent.infrastructure.gis_backends.gdal_ops.calculate_pixel_statistics import calculate_pixel_statistics
from geospatial_agent.infrastructure.gis_backends.gdal_ops.read_shapefile import read_shapefile_into_geodataframe
from geospatial_agent.infrastructure.gis_backends.gdal_ops.read_raster_metadata import read_raster_metadata # NEW IMPORT

# Initialize DataLoader globally or pass it. For simplicity in tools, we'll initialize here.
# Using a fixed directory for tool outputs for now.
_data_loader = DataLoader(data_storage_dir="data/tool_outputs")

def reproject_raster_tool(params: ReprojectRasterParams) -> DataReprojectionResult:
    """
    Reprojects a raster dataset to a new Coordinate Reference System (CRS).

    This tool utilizes the DataLoader's processing capabilities to perform
    the raster reprojection.

    Args:
        params (ReprojectRasterParams): Pydantic model containing all necessary parameters.

    Returns:
        DataReprojectionResult: The result of the reprojection operation.
    """
    logger.info(f"Tool: Reprojecting raster '{params.input_raster.path}' to '{params.output_path}' with CRS '{params.target_crs}'.")
    result = _data_loader.process_data(
        input_path=params.input_raster.path,
        operation_type="reproject_raster",
        output_filename=Path(params.output_path).name, # Extract filename for DataLoader
        target_crs=params.target_crs.value, # Pass enum value
        resampling_method=params.resampling_method.value # Pass enum value
    )
    # The process_data method returns a DataReprojectionResult or a dict if error
    if isinstance(result, DataReprojectionResult):
        logger.success(f"Raster reprojection completed successfully: {result.reprojected_path}")
        return result
    else: # It's an error dict
        logger.error(f"Raster reprojection failed: {result.get('message', 'Unknown error')}")
        return DataReprojectionResult(
            original_path=params.input_raster.path,
            reprojected_path="N/A",
            original_crs=params.input_raster.crs or "Unknown",
            new_crs=params.target_crs.value,
            success=False,
            message=result.get('message', 'Failed to reproject raster.')
        )

def convert_vector_format_tool(params: ConvertVectorFormatParams) -> DataConversionResult:
    """
    Converts a vector dataset from one format to another.

    This tool utilizes the DataLoader's processing capabilities to perform
    the vector format conversion.

    Args:
        params (ConvertVectorFormatParams): Pydantic model containing all necessary parameters.

    Returns:
        DataConversionResult: The result of the conversion operation.
    """
    logger.info(f"Tool: Converting vector '{params.input_vector.path}' to '{params.output_format.value}' at '{params.output_path}'.")
    result = _data_loader.process_data(
        input_path=params.input_vector.path,
        operation_type="convert_vector",
        output_filename=Path(params.output_path).name, # Extract filename
        output_format=params.output_format.value # Pass enum value
    )
    if isinstance(result, DataConversionResult):
        logger.success(f"Vector conversion completed successfully: {result.converted_path}")
        return result
    else: # It's an error dict
        logger.error(f"Vector conversion failed: {result.get('message', 'Unknown error')}")
        return DataConversionResult(
            original_path=params.input_vector.path,
            converted_path="N/A",
            original_format=Path(params.input_vector.path).suffix.lstrip('.') or "Unknown",
            new_format=output_format,
            success=False,
            message=result.get('message', 'Failed to convert vector format.')
        )

def clip_raster_tool(params: ClipRasterParams) -> RasterData:
    """
    Clips a raster dataset using the geometries from a vector file.

    This tool utilizes the DataLoader's processing capabilities to perform
    the raster clipping.

    Args:
        params (ClipRasterParams): Pydantic model containing all necessary parameters.

    Returns:
        RasterData: The metadata of the clipped raster.
    """
    logger.info(f"Tool: Clipping raster '{params.input_raster.path}' with vector '{params.clipping_vector.path}' to '{params.output_path}'.")
    result = _data_loader.process_data(
        input_path=params.input_raster.path,
        operation_type="clip_raster",
        output_filename=Path(params.output_path).name, # Extract filename
        vector_clip_path=params.clipping_vector.path
    )
    if isinstance(result, RasterData):
        logger.success(f"Raster clipping completed successfully: {result.path}")
        return result
    else: # It's an error dict
        logger.error(f"Raster clipping failed: {result.get('message', 'Unknown error')}")
        # Return an empty RasterData with error info
        return RasterData(
            path="N/A",
            data_type=GeospatialDataType.RASTER, # Needs to be explicitly set for Literal
            crs="Unknown",
            metadata={"error": result.get('message', 'Failed to clip raster.')}
        )

def calculate_pixel_statistics_tool(params: CalculatePixelStatisticsParams) -> Dict[str, Any]:
    """
    Calculates basic pixel statistics (mean, standard deviation, min, max)
    for a specified band of a raster dataset.

    This tool utilizes the DataLoader's processing capabilities.

    Args:
        params (CalculatePixelStatisticsParams): Pydantic model containing all necessary parameters.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated statistics.
    """
    logger.info(f"Tool: Calculating pixel statistics for band {params.band_number} of raster '{params.input_raster.path}'.")
    result = _data_loader.process_data(
        input_path=params.input_raster.path,
        operation_type="calculate_pixel_statistics",
        band_number=params.band_number
    )
    if result.get("error"):
        logger.error(f"Pixel statistics calculation failed: {result.get('error')}")
    else:
        logger.success(f"Pixel statistics calculated successfully for '{params.input_raster.path}'.")
    return result

def read_vector_info_tool(params: ReadVectorInfoParams) -> VectorData:
    """
    Reads a vector file and extracts key metadata, returning it as a Pydantic VectorData model.

    This tool utilizes the DataLoader's processing capabilities.

    Args:
        params (ReadVectorInfoParams): Pydantic model containing the path to the input vector file.

    Returns:
        VectorData: A Pydantic model containing extracted metadata about the vector file.
    """
    logger.info(f"Tool: Reading vector information from '{params.input_vector}'.")
    result = _data_loader.process_data(
        input_path=params.input_vector,
        operation_type="read_vector_info"
    )
    if isinstance(result, VectorData):
        if result.metadata and "error" in result.metadata:
            logger.error(f"Reading vector info failed: {result.metadata['error']}")
        else:
            logger.success(f"Vector information read successfully from '{params.input_vector}'.")
        return result
    else: # It's an error dict
        logger.error(f"Reading vector info failed: {result.get('message', 'Unknown error')}")
        # Return an empty VectorData with error info
        return VectorData(
            path="N/A",
            data_type=GeospatialDataType.VECTOR, # Corrected from RASTER
            crs="Unknown",
            layer_name="N/A",
            geometry_type="N/A",
            feature_count=0,
            metadata={"error": result.get('message', 'Failed to read vector info.')}
        )

# NEW TOOL: Read Raster Info Tool
def read_raster_info_tool(params: ReadRasterInfoParams) -> RasterData:
    """
    Reads a raster file and extracts key metadata, returning it as a Pydantic RasterData model.

    This tool directly calls the underlying gdal_ops function.

    Args:
        params (ReadRasterInfoParams): Pydantic model containing the path to the input raster file.

    Returns:
        RasterData: A Pydantic model containing extracted metadata about the raster file.
    """
    logger.info(f"Tool: Reading raster information from '{params.input_raster}'.")
    result = read_raster_metadata(input_raster_path=params.input_raster) # Direct call to gdal_ops function

    if result.metadata and "error" in result.metadata:
        logger.error(f"Reading raster info failed: {result.metadata['error']}")
    else:
        logger.success(f"Raster information read successfully from '{params.input_raster}'.")
    return result


def download_bhoonidhi_data_tool(params: DownloadBhoonidhiDataParams) -> DataDownloadResult:
    """
    Downloads geospatial data from the Bhoonidhi platform.

    Args:
        params (DownloadBhoonidhiDataParams): Pydantic model containing download parameters.

    Returns:
        DataDownloadResult: The result of the download operation.
    """
    logger.info(f"Tool: Downloading {params.dataset_type} data from Bhoonidhi for AOI.")
    result = _data_loader.download_bhoonidhi_data(
        aoi_geojson=params.aoi_geojson,
        dataset_type=params.dataset_type,
        output_filename=params.output_filename,
        bhoonidhi_api_key=params.bhoonidhi_api_key
    )
    if result.path == "N/A" and result.description and "Failed" in result.description:
        logger.error(f"Bhoonidhi download failed: {result.description}")
    else:
        logger.success(f"Bhoonidhi download completed: {result.path}")
    return result

def download_osm_data_tool(params: DownloadOSMDataParams) -> DataDownloadResult:
    """
    Downloads OpenStreetMap (OSM) data using the Overpass API.

    Args:
        params (DownloadOSMDataParams): Pydantic model containing download parameters.

    Returns:
        DataDownloadResult: The result of the download operation.
    """
    logger.info(f"Tool: Downloading OSM '{params.amenity_type}' data for bbox {params.bbox}.")
    result = _data_loader.download_osm_data(
        bbox=params.bbox,
        amenity_type=params.amenity_type,
        output_filename=params.output_filename,
        overpass_url=params.overpass_url
    )
    if result.path == "N/A" and result.description and "Failed" in result.description:
        logger.error(f"OSM download failed: {result.description}")
    else:
        logger.success(f"OSM download completed: {result.path}")
    return result
