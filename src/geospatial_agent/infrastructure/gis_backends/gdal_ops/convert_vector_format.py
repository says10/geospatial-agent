# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/convert_vector_format.py

import geopandas as gpd
from pathlib import Path
from typing import Union
from loguru import logger # For logging errors and important events

# Assuming geospatial_agent is installed or reachable in sys.path
from geospatial_agent.domain.schemas import DataConversionResult

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def convert_vector_format(
    input_vector_path: Union[str, Path],
    output_vector_path: Union[str, Path],
    output_format: str, # e.g., "GeoJSON", "ESRI Shapefile", "GPKG", "FlatGeobuf"
) -> DataConversionResult:
    """
    Converts a vector dataset from one format to another using GeoPandas (which uses Fiona/GDAL).

    This function supports a wide range of vector formats as long as the underlying
    Fiona/GDAL drivers are available. Common formats include "GeoJSON", "ESRI Shapefile",
    "GPKG" (GeoPackage), "FlatGeobuf", etc.

    Args:
        input_vector_path (Union[str, Path]): Path to the input vector file.
        output_vector_path (Union[str, Path]): Path for the converted output vector file.
                                                Note: For Shapefiles, this should be the path
                                                to the .shp file, but a directory will be created.
        output_format (str): The target format (e.g., "GeoJSON", "ESRI Shapefile", "GPKG").
                             Refer to Fiona's supported drivers for a complete list.

    Returns:
        DataConversionResult: A Pydantic model containing the result of the conversion.
                              Includes original/converted paths, formats, success status, and a message.
                              If an error occurs, success will be False and an error message provided.
    """
    input_vector_path = Path(input_vector_path)
    output_vector_path = Path(output_vector_path)

    logger.info(f"Attempting to convert vector data from '{input_vector_path}' to '{output_vector_path}' in format '{output_format}'.")

    try:
        # Read the input vector file into a GeoDataFrame
        gdf = gpd.read_file(input_vector_path)
        logger.debug(f"Successfully read input vector file. CRS: {gdf.crs}, Features: {len(gdf)}")

        # Ensure output directory exists for formats like Shapefile which are folders
        if output_format.lower() == "esri shapefile":
            output_vector_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_vector_path.parent.mkdir(parents=True, exist_ok=True)


        # Write the GeoDataFrame to the specified output format
        gdf.to_file(output_vector_path, driver=output_format)
        logger.success(f"Vector data successfully converted to '{output_vector_path}'.")

        return DataConversionResult(
            original_path=str(input_vector_path),
            converted_path=str(output_vector_path),
            original_format=input_vector_path.suffix.lstrip('.'), # Get extension without dot
            new_format=output_format,
            success=True,
            message=f"Vector data successfully converted from {input_vector_path.suffix.lstrip('.')} to {output_format}.",
        )
    except FileNotFoundError:
        logger.error(f"Input vector file not found: {input_vector_path}")
        return DataConversionResult(
            original_path=str(input_vector_path),
            converted_path="N/A",
            original_format=input_vector_path.suffix.lstrip('.'),
            new_format=output_format,
            success=False,
            message=f"Error: Input vector file not found at '{input_vector_path}'.",
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during vector format conversion: {e}")
        return DataConversionResult(
            original_path=str(input_vector_path),
            converted_path="N/A", # Cannot guarantee path if conversion failed early
            original_format=input_vector_path.suffix.lstrip('.'),
            new_format=output_format,
            success=False,
            message=f"An unexpected error occurred during conversion: {e}",
        )
