import os
import sys
from pathlib import Path

# Define the base directory of the project
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
GEOSPATIAL_AGENT_DIR = SRC_DIR / "geospatial_agent"

# Add src directory to sys.path for module imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added '{SRC_DIR}' to sys.path.")
else:
    print(f"'{SRC_DIR}' already in sys.path.")

print("\n--- Current sys.path ---")
for p in sys.path:
    print(p)
print("------------------------\n")


def ensure_directory(path: Path):
    """Ensures a directory exists and creates an __init__.py file."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exist: {path}")

    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created __init__.py: {init_file}")
    else:
        print(f"__init__.py already exist: {init_file}")


def write_file_content(file_path: Path, content: str):
    """Writes content to a specified file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote content to: {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def create_dummy_geotiff(filepath: Path, width=100, height=100):
    """Creates a dummy GeoTIFF file for testing."""
    try:
        import rasterio
        import numpy as np

        if filepath.exists():
            print(f"Dummy raster already exists: {filepath}")
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": "EPSG:4326",  # WGS84
            "transform": rasterio.transform.from_bounds(72, 12, 75, 15, width, height),
            "nodata": -9999.0,
        }
        with rasterio.open(filepath, "w", **profile) as dst:
            data = np.random.rand(height, width).astype(rasterio.float32) * 1000
            dst.write(data, 1)
        print(f"Created dummy raster: {filepath}")
    except ImportError:
        print("rasterio and numpy not installed. Skipping dummy GeoTIFF creation.")
    except Exception as e:
        print(f"Error creating dummy GeoTIFF: {e}")


def create_dummy_geojson(filepath: Path):
    """Creates a dummy GeoJSON file for testing."""
    try:
        import json
        if filepath.exists():
            print(f"Dummy vector already exists: {filepath}")
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)

        geojson_content = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Dummy AOI"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [73.0, 13.0],
                                [73.0, 14.0],
                                [74.0, 14.0],
                                [74.0, 13.0],
                                [73.0, 13.0],
                            ]
                        ],
                    },
                }
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson_content, f, indent=2)
        print(f"Created dummy vector: {filepath}")
    except ImportError:
        print("json module is always available. Check other issues.")
    except Exception as e:
        print(f"Error creating dummy GeoJSON: {e}")


def setup_project():
    print("--- Starting Automated Project Setup ---")

    # 1. Create core directories and __init__.py files
    print("\n--- 1. Creating core directories and __init__.py files ---")
    ensure_directory(GEOSPATIAL_AGENT_DIR)
    ensure_directory(GEOSPATIAL_AGENT_DIR / "domain")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "infrastructure")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "infrastructure" / "tools")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "data_ingestion")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "ontology")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "tool_manager")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "application")
    ensure_directory(GEOSPATIAL_AGENT_DIR / "application" / "agents")
    print("All necessary directories and __init__.py files ensured.")

    # 2. Define file contents
    print("\n--- 2. Writing content to Python files ---")

    # src/geospatial_agent/domain/schemas.py
    schemas_content = """# src/geospatial_agent/domain/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RasterData(BaseModel):
    \"\"\"
    Represents raster data with essential metadata.

    Attributes:
        path (str): File path to the raster data.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326', 'ESRI:102008').
        width (int): Width of the raster in pixels.
        height (int): Height of the raster in pixels.
        pixel_size_x (float): Pixel size in the X direction (georeferenced units).
        pixel_size_y (float): Pixel size in the Y direction (georeferenced units).
        band_count (int): Number of bands in the raster.
        no_data_value (Optional[float]): NoData value if present in the raster. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata as a dictionary. Defaults to None.
    \"\"\"
    path: str = Field(..., description="File path to the raster data.")
    crs: str = Field(..., description="Coordinate Reference System (e.g., 'EPSG:4326').")
    width: int = Field(..., description="Width of the raster in pixels.")
    height: int = Field(..., description="Height of the raster in pixels.")
    pixel_size_x: float = Field(..., description="Pixel size in X direction (georeferenced units).")
    pixel_size_y: float = Field(..., description="Pixel size in Y direction (georeferenced units).")
    band_count: int = Field(..., description="Number of bands in the raster.")
    no_data_value: Optional[float] = Field(None, description="NoData value if present.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as a dictionary.")

class VectorData(BaseModel):
    \"\"\"
    Represents vector data with essential metadata.

    Attributes:
        path (str): File path to the vector data.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326').
        layer_name (str): Name of the primary layer within the vector file.
        geometry_type (Optional[str]): Type of geometry (e.g., 'Polygon', 'Point', 'LineString').
                                       Defaults to None.
        feature_count (Optional[int]): Number of features in the vector layer. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata as a dictionary. Defaults to None.
    \"\"\"
    path: str = Field(..., description="File path to the vector data.")
    crs: str = Field(..., description="Coordinate Reference System (e.g., 'EPSG:4326').")
    layer_name: str = Field(..., description="Name of the primary layer within the vector file.")
    geometry_type: Optional[str] = Field(None, description="Type of geometry (e.g., 'Polygon', 'Point', 'LineString').")
    feature_count: Optional[int] = Field(None, description="Number of features in the vector layer.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as a dictionary.")

class AreaCalculationResult(BaseModel):
    \"\"\"
    Represents the result of an area calculation, potentially with slope correction.

    Attributes:
        area_sq_m (float): Calculated area in square meters.
        unit (str): Unit of the calculated area. Defaults to "square meters".
        notes (Optional[str]): Any additional notes about the calculation (e.g., 'slope corrected').
                               Defaults to None.
    \"\"\"
    area_sq_m: float = Field(..., description="Calculated area in square meters.")
    unit: str = Field("square meters", description="Unit of the calculated area.")
    notes: Optional[str] = Field(None, description="Any additional notes about the calculation (e.g., 'slope corrected').")

class DataDownloadResult(BaseModel):
    \"\"\"
    Represents the result of downloading data.

    Attributes:
        path (str): Local file path to the downloaded data.
        data_source (str): Source from which the data was downloaded (e.g., 'Bhoonidhi', 'OpenStreetMap').
        original_url (Optional[str]): Original URL if downloaded from a web source. Defaults to None.
        file_type (str): Type of the downloaded file (e.g., 'GeoTIFF', 'Shapefile', 'GeoJSON').
        description (Optional[str]): A brief description of the downloaded data. Defaults to None.
    \"\"\"
    path: str = Field(..., description="Local file path to the downloaded data.")
    data_source: str = Field(..., description="Source from which the data was downloaded (e.g., 'Bhoonidhi', 'OpenStreetMap').")
    original_url: Optional[str] = Field(None, description="Original URL if downloaded from a web source.")
    file_type: str = Field(..., description="Type of the downloaded file (e.g., 'GeoTIFF', 'Shapefile', 'GeoJSON').")
    description: Optional[str] = Field(None, description="A brief description of the downloaded data.")

class DataConversionResult(BaseModel):
    \"\"\"
    Represents the result of converting data from one format to another.

    Attributes:
        original_path (str): Path to the original data file.
        converted_path (str): Path to the newly converted data file.
        original_format (str): Original format of the data (e.g., 'Shapefile', 'GeoTIFF').
        new_format (str): New format of the data (e.g., 'GeoJSON', 'COG').
        success (bool): True if conversion was successful, False otherwise.
        message (Optional[str]): Details about the conversion process or any errors. Defaults to None.
    \"\"\"
    original_path: str = Field(..., description="Path to the original data file.")
    converted_path: str = Field(..., description="Path to the newly converted data file.")
    original_format: str = Field(..., description="Original format of the data (e.g., 'Shapefile', 'GeoTIFF').")
    new_format: str = Field(..., description="New format of the data (e.g., 'GeoJSON', 'COG').")
    success: bool = Field(..., description="True if conversion was successful, False otherwise.")
    message: Optional[str] = Field(None, description="Details about the conversion process or any errors.")

class DataReprojectionResult(BaseModel):
    \"\"\"
    Represents the result of reprojecting geospatial data.

    Attributes:
        original_path (str): Path to the original data file.
        reprojected_path (str): Path to the newly reprojected data file.
        original_crs (str): Original Coordinate Reference System (e.g., 'EPSG:4326').
        new_crs (str): New Coordinate Reference System (e.g., 'EPSG:3857').
        success (bool): True if reprojection was successful, False otherwise.
        message (Optional[str]): Details about the reprojection process or any errors. Defaults to None.
    \"\"\"
    original_path: str = Field(..., description="Path to the original data file.")
    reprojected_path: str = Field(..., description="Path to the newly reprojected data file.")
    original_crs: str = Field(..., description="Original Coordinate Reference System (e.g., 'EPSG:4326').")
    new_crs: str = Field(..., description="New Coordinate Reference System (e.g., 'EPSG:3857').")
    success: bool = Field(..., description="True if reprojection was successful, False otherwise.")
    message: Optional[str] = Field(None, description="Details about the reprojection process or any errors.")
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "domain" / "schemas.py", schemas_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/reproject_raster.py
    reproject_raster_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/reproject_raster.py

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
from typing import Union
from loguru import logger # For logging errors and important events
import os # Import os for cpu_count

# Assuming geospatial_agent is installed or reachable in sys.path
from geospatial_agent.domain.schemas import DataReprojectionResult

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def reproject_raster(
    input_raster_path: Union[str, Path],
    output_raster_path: Union[str, Path],
    target_crs: str,
    resampling_method: Resampling = Resampling.nearest,
) -> DataReprojectionResult:
    \"\"\"
    Reprojects a raster dataset to a new Coordinate Reference System (CRS).

    This function leverages rasterio's capabilities to perform a robust reprojection,
    handling multiple bands and calculating appropriate output dimensions and transform.

    Args:
        input_raster_path (Union[str, Path]): Path to the input raster file.
        output_raster_path (Union[str, Path]): Path for the reprojected output raster file.
        target_crs (str): The target CRS (e.g., 'EPSG:4326', 'ESRI:102008', 'PROJ:32610').
                          Can be an EPSG code, WKT string, or Proj4 string.
        resampling_method (Resampling): The resampling algorithm to use when transforming
                                        pixel values. Options include Resampling.nearest
                                        (default, fastest), Resampling.bilinear,
                                        Resampling.cubic, etc.

    Returns:
        DataReprojectionResult: A Pydantic model containing the result of the reprojection.
                                Includes paths, original/new CRSs, success status, and a message.
                                If an error occurs, success will be False and an error message provided.
    \"\"\"
    input_raster_path = Path(input_raster_path)
    output_raster_path = Path(output_raster_path)

    logger.info(f"Attempting to reproject raster from '{input_raster_path}' to '{output_raster_path}' with target CRS '{target_crs}'.")

    try:
        with rasterio.open(input_raster_path) as src:
            original_crs_str = str(src.crs)
            logger.debug(f"Input raster CRS: {original_crs_str}")

            # Calculate the transform and dimensions for the reprojected dataset
            # This ensures the output raster covers the necessary area and has correct resolution
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            logger.debug(f"Calculated output dimensions: width={width}, height={height}")

            # Update metadata for the output raster
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "driver": "GTiff", # Ensure output is GeoTIFF
                    "nodata": src.nodata # Preserve NoData value
                }
            )
            logger.debug(f"Output raster metadata: {kwargs}")

            # Write the reprojected raster
            with rasterio.open(output_raster_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1): # Iterate through all bands
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=resampling_method,
                        num_threads=os.cpu_count() # Use all available CPU cores for faster processing
                    )
            logger.success(f"Raster successfully reprojected to '{output_raster_path}'.")
            return DataReprojectionResult(
                original_path=str(input_raster_path),
                reprojected_path=str(output_raster_path),
                original_crs=original_crs_str,
                new_crs=target_crs,
                success=True,
                message=f"Raster successfully reprojected from {original_crs_str} to {target_crs}.",
            )
    except FileNotFoundError:
        logger.error(f"Input raster file not found: {input_raster_path}")
        return DataReprojectionResult(
            original_path=str(input_raster_path),
            reprojected_path="N/A",
            original_crs="Unknown",
            new_crs=target_crs,
            success=False,
            message=f"Error: Input raster file not found at '{input_raster_path}'.",
        )
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Raster I/O error for '{input_raster_path}': {e}")
        return DataReprojectionResult(
            original_path=str(input_raster_path),
            reprojected_path="N/A",
            original_crs="Unknown",
            new_crs=target_crs,
            success=False,
            message=f"Raster I/O error: {e}",
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during raster reprojection: {e}")
        return DataReprojectionResult(
            original_path=str(input_raster_path),
            reprojected_path="N/A",
            original_crs="Unknown",
            new_crs=target_crs,
            success=False,
            message=f"An unexpected error occurred: {e}",
        )
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "reproject_raster.py", reproject_raster_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/convert_vector_format.py
    convert_vector_format_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/convert_vector_format.py

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
    \"\"\"
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
    \"\"\"
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
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "convert_vector_format.py", convert_vector_format_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/clip_raster.py
    clip_raster_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/clip_raster.py

import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from pathlib import Path
from typing import Union
from loguru import logger # For logging errors and important events

# Assuming geospatial_agent is installed or reachable in sys.path
from geospatial_agent.domain.schemas import RasterData

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def clip_raster_with_vector(
    input_raster_path: Union[str, Path],
    input_vector_path: Union[str, Path],
    output_clipped_raster_path: Union[str, Path],
) -> RasterData:
    \"\"\"
    Clips a raster dataset using the geometries from a vector file.

    This function reads the vector geometries, reprojects them to the raster's CRS
    if necessary, and then uses rasterio's masking capabilities to clip the raster.
    The output raster will have the same CRS and properties as the input raster,
    but its extent will be limited by the clipping geometries.

    Args:
        input_raster_path (Union[str, Path]): Path to the input raster file.
        input_vector_path (Union[str, Path]): Path to the vector file containing clipping geometries.
                                              Supported formats include Shapefile, GeoJSON, GPKG, etc.
        output_clipped_raster_path (Union[str, Path]): Path for the clipped output raster file.

    Returns:
        RasterData: A Pydantic model representing the clipped raster data with its metadata.
                    If clipping fails, an empty RasterData model will be returned with
                    an error message in its metadata.
    \"\"\"
    input_raster_path = Path(input_raster_path)
    input_vector_path = Path(input_vector_path)
    output_clipped_raster_path = Path(output_clipped_raster_path)

    logger.info(f"Attempting to clip raster '{input_raster_path}' with vector '{input_vector_path}'.")

    try:
        # Read the vector data to get the geometries
        gdf = gpd.read_file(input_vector_path)
        logger.debug(f"Read vector data. CRS: {gdf.crs}, Features: {len(gdf)}")

        # Open the raster dataset
        with rasterio.open(input_raster_path) as src:
            logger.debug(f"Opened raster data. CRS: {src.crs}, Bounds: {src.bounds}")

            # Check if CRSs match; reproject vector if necessary
            if src.crs != gdf.crs:
                logger.warning(f"CRS mismatch: Raster is {src.crs}, Vector is {gdf.crs}. Reprojecting vector to raster CRS for clipping.")
                gdf_reprojected = gdf.to_crs(src.crs)
                geometries = [mapping(geom) for geom in gdf_reprojected.geometry]
            else:
                geometries = [mapping(geom) for geom in gdf.geometry]

            # Perform the clipping operation
            out_image, out_transform = mask(src, geometries, crop=True, nodata=src.nodata)
            logger.debug(f"Raster masked. Output image shape: {out_image.shape}")

            # Update metadata for the clipped raster
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs, # CRS remains the same as original raster
                "nodata": src.nodata # Ensure nodata is preserved
            })
            logger.debug(f"Clipped raster metadata: {out_meta}")

            # Ensure output directory exists
            output_clipped_raster_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the clipped raster to file
            with rasterio.open(output_clipped_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)
            logger.success(f"Raster successfully clipped and saved to '{output_clipped_raster_path}'.")

            # Construct and return RasterData from the newly created clipped file
            with rasterio.open(output_clipped_raster_path) as clipped_src:
                return RasterData(
                    path=str(output_clipped_raster_path),
                    crs=str(clipped_src.crs),
                    width=clipped_src.width,
                    height=clipped_src.height,
                    pixel_size_x=clipped_src.res[0],
                    pixel_size_y=clipped_src.res[1],
                    band_count=clipped_src.count,
                    no_data_value=clipped_src.nodata,
                    metadata=clipped_src.tags(),
                )

    except FileNotFoundError as e:
        logger.error(f"File not found during clipping: {e}")
        return RasterData(
            path="N/A",
            crs="Unknown",
            width=0, height=0, pixel_size_x=0.0, pixel_size_y=0.0, band_count=0,
            metadata={"error": str(e), "message": "Input file not found for clipping."},
        )
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Raster I/O error during clipping: {e}")
        return RasterData(
            path="N/A",
            crs="Unknown",
            width=0, height=0, pixel_size_x=0.0, pixel_size_y=0.0, band_count=0,
            metadata={"error": str(e), "message": "Raster I/O error during clipping."},
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during raster clipping: {e}")
        return RasterData(
            path="N/A",
            crs="Unknown",
            width=0, height=0, pixel_size_x=0.0, pixel_size_y=0.0, band_count=0,
            metadata={"error": str(e), "message": f"An unexpected error occurred: {e}"},
        )
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "clip_raster.py", clip_raster_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/calculate_pixel_statistics.py
    calculate_pixel_statistics_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/calculate_pixel_statistics.py

import rasterio
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
from loguru import logger # For logging errors and important events

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def calculate_pixel_statistics(
    input_raster_path: Union[str, Path],
    band_number: int = 1,
) -> Dict[str, Any]:
    \"\"\"
    Calculates basic pixel statistics (mean, standard deviation, min, max)
    for a specified band of a raster dataset.

    This function handles NoData values by masking them out before calculating statistics,
    ensuring that statistics are computed only on valid data.

    Args:
        input_raster_path (Union[str, Path]): Path to the input raster file.
        band_number (int): The band number to calculate statistics for (1-indexed). Defaults to 1.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated statistics ('mean', 'std', 'min', 'max'),
                        along with 'band_number', 'raster_path', and an 'error' message if any.
                        If an error occurs or no valid data is found, relevant fields will be None or contain error info.
    \"\"\"
    input_raster_path = Path(input_raster_path)

    stats: Dict[str, Any] = {
        "raster_path": str(input_raster_path),
        "band_number": band_number,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "error": None,
    }

    logger.info(f"Calculating pixel statistics for band {band_number} of raster '{input_raster_path}'.")

    try:
        with rasterio.open(input_raster_path) as src:
            if band_number > src.count or band_number < 1:
                error_msg = f"Band number {band_number} is out of range. Raster has {src.count} bands."
                logger.error(error_msg)
                stats["error"] = error_msg
                return stats

            # Read the specified band data
            band_data = src.read(band_number)
            logger.debug(f"Read band {band_number} data with shape: {band_data.shape}")

            # Handle NoData values by masking them out
            if src.nodata is not None:
                logger.debug(f"NoData value detected: {src.nodata}. Masking data.")
                masked_data = np.ma.masked_equal(band_data, src.nodata)
                if masked_data.count() == 0: # Check if all data is NoData after masking
                    error_msg = "All pixels in the specified band are NoData. Cannot calculate statistics."
                    logger.warning(error_msg)
                    stats["error"] = error_msg
                    return stats
                band_data = masked_data.compressed() # Use only valid (unmasked) data for statistics
            elif band_data.size == 0: # Handle case of an empty band (e.g., from a clipped zero-size raster)
                error_msg = "Band data is empty. Cannot calculate statistics."
                logger.warning(error_msg)
                stats["error"] = error_msg
                return stats

            # Calculate statistics on the valid data
            stats["mean"] = float(np.mean(band_data))
            stats["std"] = float(np.std(band_data))
            stats["min"] = float(np.min(band_data))
            stats["max"] = float(np.max(band_data))
            logger.success(f"Statistics calculated: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}.")

    except FileNotFoundError:
        error_msg = f"Raster file not found: {input_raster_path}"
        logger.error(error_msg)
        stats["error"] = error_msg
    except rasterio.errors.RasterioIOError as e:
        error_msg = f"Raster I/O error for '{input_raster_path}': {e}"
        logger.error(error_msg)
        stats["error"] = error_msg
    except ValueError as e:
        error_msg = str(e) # Catching specific ValueErrors like band out of range
        logger.error(error_msg)
        stats["error"] = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during pixel statistics calculation: {e}"
        logger.error(error_msg)
        stats["error"] = error_msg

    return stats
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "calculate_pixel_statistics.py", calculate_pixel_statistics_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_shapefile.py
    read_shapefile_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_shapefile.py

import geopandas as gpd
from pathlib import Path
from typing import Union
from loguru import logger # For logging errors and important events

# Assuming geospatial_agent is installed or reachable in sys.path
from geospatial_agent.domain.schemas import VectorData

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def read_shapefile_into_geodataframe(
    input_vector_path: Union[str, Path]
) -> VectorData:
    \"\"\"
    Reads a vector file (e.g., Shapefile, GeoJSON, GPKG) into a GeoDataFrame
    and extracts key metadata, returning it as a Pydantic VectorData model.

    This function provides a structured way to get information about a vector dataset
    without exposing the full GeoDataFrame object directly, adhering to the
    principle of returning Pydantic models for structured data.

    Args:
        input_vector_path (Union[str, Path]): Path to the input vector file.
                                              Supported formats include .shp, .geojson, .gpkg, etc.

    Returns:
        VectorData: A Pydantic model containing extracted metadata about the vector file.
                    Includes path, CRS, layer name, geometry type, feature count, and additional metadata.
                    If reading fails, an empty VectorData model will be returned with
                    an error message in its metadata.
    \"\"\"
    input_vector_path = Path(input_vector_path)

    logger.info(f"Attempting to read vector file and extract metadata from '{input_vector_path}'.")

    try:
        # Read the vector file into a GeoDataFrame
        gdf = gpd.read_file(input_vector_path)
        logger.debug(f"Successfully read vector file. Features: {len(gdf)}")

        # Determine geometry type:
        # Iterate through geometries to find the first non-empty type.
        # If all are empty, or no features, set to a default.
        geometry_type = "Unknown"
        if not gdf.empty:
            # Find the first non-empty geometry to determine its type
            non_empty_geoms = gdf.geometry.dropna()
            if not non_empty_geoms.empty:
                geometry_type = non_empty_geoms.iloc[0].geom_type
            else:
                geometry_type = "No valid geometries found"
        else:
            geometry_type = "No features found"

        # Extract layer name from the file name (stem)
        layer_name = input_vector_path.stem
        logger.debug(f"Extracted layer name: '{layer_name}', Geometry type: '{geometry_type}'")

        # Prepare additional metadata (e.g., column names, file size)
        additional_metadata = {
            "columns": list(gdf.columns),
            "file_size_bytes": input_vector_path.stat().st_size if input_vector_path.exists() else 0,
            "driver": gdf.crs.to_wkt() if gdf.crs else "Unknown" # Store CRS as WKT in metadata for completeness
        }
        logger.debug(f"Additional metadata: {additional_metadata}")

        logger.success(f"Successfully read vector data from '{input_vector_path}'.")
        return VectorData(
            path=str(input_vector_path),
            crs=str(gdf.crs),
            layer_name=layer_name,
            geometry_type=geometry_type,
            feature_count=len(gdf),
            metadata=additional_metadata,
        )
    except FileNotFoundError:
        error_msg = f"Vector file not found: {input_vector_path}"
        logger.error(error_msg)
        return VectorData(
            path=str(input_vector_path),
            crs="Unknown",
            layer_name="N/A",
            geometry_type="N/A",
            feature_count=0,
            metadata={"error": error_msg, "message": "File not found."},
        )
    except Exception as e:
        error_msg = f"An unexpected error occurred while reading vector file '{input_vector_path}': {e}"
        logger.error(error_msg)
        return VectorData(
            path=str(input_vector_path),
            crs="Unknown",
            layer_name="N/A",
            geometry_type="N/A",
            feature_count=0,
            metadata={"error": str(e), "message": "Failed to read vector file."},
        )
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "read_shapefile.py", read_shapefile_content)

    # src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_raster_metadata.py
    # Corrected content for read_raster_metadata.py
    read_raster_metadata_content = """# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_raster_metadata.py

from pathlib import Path
from typing import Union
from loguru import logger
import rasterio
from rasterio.errors import RasterioIOError

# Corrected import: GeospatialDataType is in ontology.ontology, not domain.schemas
from geospatial_agent.domain.schemas import RasterData
from geospatial_agent.ontology.ontology import GeospatialDataType # Corrected import

def read_raster_metadata(
    input_raster_path: Union[str, Path]
) -> RasterData:
    \"\"\"
    Reads a raster file and extracts key metadata (CRS, dimensions, pixel size, band count).

    Args:
        input_raster_path (Union[str, Path]): Path to the input raster file.

    Returns:
        RasterData: A Pydantic model containing extracted metadata about the raster file.
    \"\"\"
    input_raster_path = Path(input_raster_path)
    logger.info(f"Attempting to read raster metadata from '{input_raster_path}'.")

    try:
        with rasterio.open(input_raster_path) as src:
            crs_str = str(src.crs) if src.crs else "Unknown"
            
            logger.success(f"Successfully read raster metadata from '{input_raster_path}'.")
            return RasterData(
                path=str(input_raster_path),
                crs=crs_str,
                width=src.width,
                height=src.height,
                pixel_size_x=src.res[0],
                pixel_size_y=src.res[1],
                band_count=src.count,
                no_data_value=src.nodata,
                data_type=GeospatialDataType.RASTER, # Explicitly set for Literal
                metadata={"message": "Raster metadata read successfully."}
            )

    except RasterioIOError as e:
        logger.error(f"Raster I/O error reading metadata from '{input_raster_path}': {e}")
        return RasterData(
            path=str(input_raster_path),
            crs="N/A",
            width=0, height=0,
            pixel_size_x=0.0, pixel_size_y=0.0,
            band_count=0,
            data_type=GeospatialDataType.RASTER,
            metadata={"error": f"Raster I/O error: {e}"}
        )
    except FileNotFoundError:
        logger.error(f"Input raster file not found: '{input_raster_path}'.")
        return RasterData(
            path=str(input_raster_path),
            crs="N/A",
            width=0, height=0,
            pixel_size_x=0.0, pixel_size_y=0.0,
            band_count=0,
            data_type=GeospatialDataType.RASTER,
            metadata={"error": f"Input file not found: {input_raster_path}"}
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading raster metadata from '{input_raster_path}': {e}")
        return RasterData(
            path=str(input_raster_path),
            crs="N/A",
            width=0, height=0,
            pixel_size_x=0.0, pixel_size_y=0.0,
            band_count=0,
            data_type=GeospatialDataType.RASTER,
            metadata={"error": f"Unexpected error: {e}"}
        )
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "infrastructure" / "gis_backends" / "gdal_ops" / "read_raster_metadata.py", read_raster_metadata_content)


    # src/geospatial_agent/data_ingestion/data_loader.py
    # Corrected content for data_loader.py
    data_loader_content = """# src/geospatial_agent/data_ingestion/data_loader.py

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from loguru import logger
import json
import requests # For simulating web downloads
import time # For simulating delays

# Import Pydantic schemas for data representation and operation results
from geospatial_agent.domain.schemas import (
    RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult
)
# Import concrete GDAL operations
from geospatial_agent.infrastructure.gis_backends.gdal_ops.reproject_raster import reproject_raster
from geospatial_agent.infrastructure.gis_backends.gdal_ops.convert_vector_format import convert_vector_format
from geospatial_agent.infrastructure.gis_backends.gdal_ops.clip_raster import clip_raster_with_vector
from geospatial_agent.infrastructure.gis_backends.gdal_ops.calculate_pixel_statistics import calculate_pixel_statistics
from geospatial_agent.infrastructure.gis_backends.gdal_ops.read_shapefile import read_shapefile_into_geodataframe
from geospatial_agent.infrastructure.gis_backends.gdal_ops.read_raster_metadata import read_raster_metadata # NEW IMPORT

class DataLoader:
    \"\"\"
    Manages the loading, downloading, and basic processing of geospatial data.

    This class acts as an interface between the core agent logic and the underlying
    geospatial data infrastructure. It can simulate data downloads and orchestrate
    calls to specific GDAL-based operations.
    \"\"\"

    def __init__(self, data_storage_dir: Union[str, Path] = "data/tool_outputs"):
        \"\"\"
        Initializes the DataLoader.

        Args:
            data_storage_dir (Union[str, Path]): The base directory where downloaded
                                                  and processed data will be stored.
        \"\"\"
        self.data_storage_dir = Path(data_storage_dir)
        self.download_dir = Path("data/downloads") # Separate dir for initial downloads
        self.data_storage_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataLoader initialized. Data will be stored in: {self.data_storage_dir}")

    def download_bhoonidhi_data(
        self,
        aoi_geojson: Dict[str, Any],
        dataset_type: str, # e.g., "DEM", "SatelliteImagery"
        output_filename: str,
        bhoonidhi_api_key: Optional[str] = None,
    ) -> DataDownloadResult:
        \"\"\"
        Simulates downloading geospatial data from the Bhoonidhi platform.

        In a real application, this would involve API calls to Bhoonidhi.
        For this simulation, it creates a dummy GeoTIFF or GeoJSON.

        Args:
            aoi_geojson (Dict[str, Any]): GeoJSON dictionary defining the Area of Interest.
            dataset_type (str): Type of dataset to download (e.g., "DEM", "SatelliteImagery").
            output_filename (str): Desired filename for the downloaded data.
            bhoonidhi_api_key (Optional[str]): API key for Bhoonidhi (ignored in simulation).

        Returns:
            DataDownloadResult: A Pydantic model representing the result of the download.
        \"\"\"
        output_path = self.data_storage_dir / output_filename
        logger.info(f"Simulating Bhoonidhi data download for '{dataset_type}' to '{output_path}'.")
        print(f"DEBUG: Entering download_bhoonidhi_data for {output_filename}") # DEBUG PRINT
        time.sleep(1) # Simulate network delay

        try:
            # Simulate different data types based on expected output
            if output_filename.lower().endswith(".tif"):
                # Create a simple empty file or a placeholder for GeoTIFF
                # The actual dummy_dem.tif is created by setup_project.py
                output_path.touch()
                file_type = "GeoTIFF"
            elif output_filename.lower().endswith(".geojson") or output_filename.lower().endswith(".shp"):
                # Create a dummy GeoJSON (or indicate shapefile creation)
                dummy_geojson_content = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"name": f"Simulated {dataset_type}"},
                            "geometry": aoi_geojson
                        }
                    ]
                }
                with open(output_path, "w") as f:
                    json.dump(dummy_geojson_content, f, indent=2)
                file_type = "GeoJSON" if output_filename.lower().endswith(".geojson") else "Shapefile (Simulated)"
            else:
                # Default to a generic text file for other types
                with open(output_path, "w") as f:
                    f.write(f"Simulated data for {dataset_type} within AOI: {aoi_geojson}")
                file_type = "TXT"

            logger.success(f"Simulated download of Bhoonidhi data ({file_type}) to '{output_path}'.")
            return DataDownloadResult(
                path=str(output_path),
                data_source="Bhoonidhi (Simulated)",
                original_url=None,
                file_type=file_type,
                description=f"Simulated {dataset_type} data for AOI.",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to simulate Bhoonidhi data download: {e}")
            return DataDownloadResult(
                path="N/A",
                data_source="Bhoonidhi (Simulated)",
                original_url=None,
                file_type="N/A",
                description=f"Failed to simulate download: {e}",
                success=False
            )

    def download_osm_data(
        self,
        bbox: List[float], # [min_lon, min_lat, max_lon, max_lat]
        amenity_type: str, # e.g., "school", "hospital"
        output_filename: str,
        overpass_url: str = "http://overpass-api.de/api/interpreter",
    ) -> DataDownloadResult:
        \"\"\"
        Simulates downloading OpenStreetMap (OSM) data using the Overpass API.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat].
            amenity_type: Type of amenity to query (e.g., "school", "hospital").
            output_filename: Desired filename for the downloaded data (e.g., "schools.geojson").
            overpass_url: URL for the Overpass API endpoint.

        Returns:
            DataDownloadResult: A Pydantic model representing the result of the download.
        \"\"\"
        output_path = self.data_storage_dir / output_filename
        logger.info(f"Simulating OSM data download for '{amenity_type}' in bbox {bbox} to '{output_path}'.")
        time.sleep(1) # Simulate network delay

        # In a real scenario, construct Overpass QL query and make an HTTP request
        # For simulation, create a dummy GeoJSON
        try:
            dummy_osm_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": f"Simulated {amenity_type} 1", "amenity": amenity_type},
                        "geometry": {"type": "Point", "coordinates": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": f"Simulated {amenity_type} 2", "amenity": amenity_type},
                        "geometry": {"type": "Point", "coordinates": [bbox[0] + 0.1, bbox[1] + 0.1]}
                    }
                ]
            }
            with open(output_path, "w") as f:
                json.dump(dummy_osm_data, f, indent=2)

            logger.success(f"Simulated download of OSM data to '{output_path}'.")
            return DataDownloadResult(
                path=str(output_path),
                data_source="OpenStreetMap (Simulated)",
                original_url=f"{overpass_url}?data=[out:json];node({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]})['amenity'='{amenity_type}'];out;",
                file_type="GeoJSON",
                description=f"Simulated OSM data for amenity type '{amenity_type}' within bbox {bbox}.",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to simulate OSM data download: {e}")
            return DataDownloadResult(
                path="N/A",
                data_source="OpenStreetMap (Simulated)",
                original_url=None,
                file_type="N/A",
                description=f"Failed to simulate download: {e}",
                success=False
            )


    def process_data(
        self,
        input_path: Union[str, Path],
        operation_type: str,
        output_filename: Optional[str] = None,
        **kwargs: Any
    ) -> Union[RasterData, VectorData, DataReprojectionResult, DataConversionResult, Dict[str, Any]]:
        \"\"\"
        Processes geospatial data using specified GDAL operations.

        Args:
            input_path: Path to the input data file.
            operation_type: The type of operation to perform (e.g., "reproject_raster",
                                  "convert_vector", "clip_raster", "calculate_pixel_statistics",
                                  "read_vector_info", "read_raster_info").
            output_filename: Desired filename for the output data, if applicable.
            **kwargs: Additional keyword arguments specific to the operation.

        Returns:
            The result of the operation, which can be a Pydantic model (RasterData, VectorData,
                DataReprojectionResult, DataConversionResult) or a dictionary for statistics.
        \"\"\"
        input_path = Path(input_path)
        output_path = None
        if output_filename:
            output_path = self.data_storage_dir / output_filename

        logger.info(f"Processing data '{input_path}' with operation '{operation_type}'. Output file: {output_path}.")

        try:
            if operation_type == "reproject_raster":
                if not output_path:
                    raise ValueError("output_filename is required for reproject_raster operation.")
                return reproject_raster(
                    input_raster_path=input_path,
                    output_raster_path=output_path,
                    target_crs=kwargs["target_crs"],
                    resampling_method=kwargs.get("resampling_method")
                )
            elif operation_type == "convert_vector":
                if not output_path:
                    raise ValueError("output_filename is required for convert_vector operation.")
                return convert_vector_format(
                    input_vector_path=input_path,
                    output_vector_path=output_path,
                    output_format=kwargs["output_format"]
                )
            elif operation_type == "clip_raster":
                if not output_path:
                    raise ValueError("output_filename is required for clip_raster operation.")
                return clip_raster_with_vector(
                    input_raster_path=input_path,
                    input_vector_path=kwargs["vector_clip_path"],
                    output_clipped_raster_path=output_path
                )
            elif operation_type == "calculate_pixel_statistics":
                return calculate_pixel_statistics(
                    input_raster_path=input_path,
                    band_number=kwargs.get("band_number", 1)
                )
            elif operation_type == "read_vector_info":
                return read_shapefile_into_geodataframe(
                    input_vector_path=input_path
                )
            elif operation_type == "read_raster_info": # NEW OPERATION TYPE
                return read_raster_metadata( # Direct call
                    input_raster_path=input_path
                )
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
        except Exception as e:
            logger.error(f"Error during data processing operation '{operation_type}' on '{input_path}': {e}")
            return {"error": True, "message": f"Operation '{operation_type}' failed: {e}"}
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "data_ingestion" / "data_loader.py", data_loader_content)

    # src/geospatial_agent/ontology/ontology.py
    ontology_content = """# src/geospatial_agent/ontology/ontology.py

from pydantic import BaseModel, Field, conlist, confloat
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum

# --- Enums for common geospatial concepts ---

class SpatialReferenceSystem(str, Enum):
    \"\"\"Commonly used Spatial Reference Systems (CRS).\"\"\"
    EPSG_4326 = "EPSG:4326" # WGS 84 Geographic (Lat/Lon)
    EPSG_3857 = "EPSG:3857" # Web Mercator
    EPSG_32644 = "EPSG:32644" # WGS 84 / UTM zone 44N (example for India)
    # Add more as needed

class RasterFormat(str, Enum):
    \"\"\"Supported Raster Formats.\"\"\"
    GEOTIFF = "GTiff"
    PNG = "PNG"
    JPEG = "JPEG"
    # Add more as needed

class VectorFormat(str, Enum):
    \"\"\"Supported Vector Formats.\"\"\"
    GEOJSON = "GeoJSON"
    SHAPEFILE = "ESRI Shapefile"
    GPKG = "GPKG" # GeoPackage
    FLATGEBUF = "FlatGeobuf"
    # Add more as needed

class ResamplingMethod(str, Enum):
    \"\"\"Resampling methods for raster operations.\"\"\"
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    # Add more as needed

class StatisticType(str, Enum):
    \"\"\"Types of statistics that can be calculated.\"\"\"
    MEAN = "mean"
    STD_DEV = "std_dev"
    MIN = "min"
    MAX = "max"
    # Add more as needed

# --- Core Ontology Models ---

class GeospatialDataType(str, Enum):
    \"\"\"Defines the high-level type of geospatial data.\"\"\"
    RASTER = "raster"
    VECTOR = "vector"

class SpatialExtent(BaseModel):
    \"\"\"Represents a geographic bounding box.\"\"\"
    min_lon: confloat(ge=-180, le=180) = Field(..., description="Minimum longitude (-180 to 180)")
    min_lat: confloat(ge=-90, le=90) = Field(..., description="Minimum latitude (-90 to 90)")
    max_lon: confloat(ge=-180, le=180) = Field(..., description="Maximum longitude (-180 to 180)")
    max_lat: confloat(ge=-90, le=90) = Field(..., description="Maximum latitude (-90 to 90)")

    class Config:
        json_schema_extra = {
            "example": {
                "min_lon": 72.0, "min_lat": 12.0, "max_lon": 75.0, "max_lat": 15.0
            }
        }

class GeospatialData(BaseModel):
    \"\"\"Abstract base model for any geospatial data.\"\"\"
    path: str = Field(..., description="File path to the geospatial data.")
    data_type: GeospatialDataType = Field(..., description="High-level type of the geospatial data (raster or vector).")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g., 'EPSG:4326').")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as key-value pairs.")

class RasterData(GeospatialData):
    \"\"\"Represents a raster dataset.\"\"\"
    data_type: Literal[GeospatialDataType.RASTER] = GeospatialDataType.RASTER
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_size_x: Optional[float] = None
    pixel_size_y: Optional[float] = None
    band_count: Optional[int] = None
    no_data_value: Optional[Union[int, float]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/data/dem.tif", "data_type": "raster", "crs": "EPSG:4326",
                "width": 1000, "height": 1000, "pixel_size_x": 0.001, "pixel_size_y": 0.001,
                "band_count": 1, "no_data_value": -9999
            }
        }

class VectorData(GeospatialData):
    \"\"\"Represents a vector dataset.\"\"\"
    data_type: Literal[GeospatialDataType.VECTOR] = GeospatialDataType.VECTOR
    layer_name: Optional[str] = None
    geometry_type: Optional[str] = None
    feature_count: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/data/cities.shp", "data_type": "vector", "crs": "EPSG:4326",
                "layer_name": "cities", "geometry_type": "Point", "feature_count": 500
            }
        }

# --- Operation/Tool Parameter Models ---

class ReprojectRasterParams(BaseModel):
    \"\"\"Parameters for raster reprojection.\"\"\"
    input_raster: RasterData = Field(..., description="The input raster data to reproject.")
    target_crs: SpatialReferenceSystem = Field(..., description="The target CRS for reprojection.")
    output_path: str = Field(..., description="Output path for the reprojected raster.")
    resampling_method: ResamplingMethod = Field(ResamplingMethod.NEAREST, description="Resampling method to use.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_raster": {"path": "/data/dem.tif", "data_type": "raster", "crs": "EPSG:4326"},
                "target_crs": "EPSG:3857",
                "output_path": "/data/dem_web_mercator.tif",
                "resampling_method": "bilinear"
            }
        }

class ConvertVectorFormatParams(BaseModel):
    \"\"\"Parameters for vector format conversion.\"\"\"
    input_vector: VectorData = Field(..., description="The input vector data to convert.")
    output_format: VectorFormat = Field(..., description="The target vector format.")
    output_path: str = Field(..., description="Output path for the converted vector file.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_vector": {"path": "/data/cities.shp", "data_type": "vector", "crs": "EPSG:4326"},
                "output_format": "GeoJSON",
                "output_path": "/data/cities.geojson"
            }
        }

class ClipRasterParams(BaseModel):
    \"\"\"Parameters for clipping a raster with a vector.\"\"\"
    input_raster: RasterData = Field(..., description="The raster data to be clipped.")
    clipping_vector: VectorData = Field(..., description="The vector data defining the clipping boundary.")
    output_path: str = Field(..., description="Output path for the clipped raster.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_raster": {"path": "/data/image.tif", "data_type": "raster", "crs": "EPSG:4326"},
                "clipping_vector": {"path": "/data/aoi.geojson", "data_type": "vector", "crs": "EPSG:4326"},
                "output_path": "/data/image_clipped.tif"
            }
        }

class CalculatePixelStatisticsParams(BaseModel):
    \"\"\"Parameters for calculating raster pixel statistics.\"\"\"
    input_raster: RasterData = Field(..., description="The raster data for which to calculate statistics.")
    band_number: Optional[int] = Field(1, description="The band number (1-indexed) to calculate statistics for.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_raster": {"path": "/data/dem.tif", "data_type": "raster", "crs": "EPSG:4326"},
                "band_number": 1
            }
        }

class ReadVectorInfoParams(BaseModel):
    \"\"\"Parameters for reading vector file information.\"\"\"
    input_vector: str = Field(..., description="Path to the input vector file.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_vector": "/data/countries.shp"
            }
        }

class ReadRasterInfoParams(BaseModel):
    \"\"\"Parameters for reading raster file information.\"\"\"
    input_raster: str = Field(..., description="Path to the input raster file.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_raster": "/data/dem.tif"
            }
        }

class DownloadBhoonidhiDataParams(BaseModel):
    \"\"\"Parameters for downloading data from Bhoonidhi.\"\"\"
    aoi_geojson: Dict[str, Any] = Field(..., description="Area of Interest as a GeoJSON dictionary.")
    dataset_type: str = Field(..., description="Type of dataset to download (e.g., 'DEM', 'SatelliteImagery').")
    output_filename: str = Field(..., description="Desired filename for the downloaded data.")
    bhoonidhi_api_key: Optional[str] = Field(None, description="API key for Bhoonidhi, if required.")

    class Config:
        json_schema_extra = {
            "example": {
                "aoi_geojson": {"type": "Polygon", "coordinates": [[[72,12],[72,15],[75,15],[75,12],[72,12]]]},
                "dataset_type": "SatelliteImagery",
                "output_filename": "bhoonidhi_image.tif"
            }
        }

class DownloadOSMDataParams(BaseModel):
    \"\"\"Parameters for downloading OpenStreetMap data via Overpass API.\"\"\"
    bbox: conlist(confloat(), min_length=4, max_length=4) = Field(..., description="Bounding box [min_lon, min_lat, max_lon, max_lat].")
    amenity_type: str = Field(..., description="Type of amenity to query (e.g., 'school', 'hospital').")
    output_filename: str = Field(..., description="Desired filename for the downloaded data (e.g., 'schools.geojson').")
    overpass_url: str = Field("http://overpass-api.de/api/interpreter", description="URL for the Overpass API endpoint.")

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [77.0, 28.0, 78.0, 29.0],
                "amenity_type": "restaurant",
                "output_filename": "delhi_restaurants.geojson"
            }
        }

# --- Ontology for Tool Definitions ---

class ToolParameter(BaseModel):
    \"\"\"Represents a single parameter for a tool.\"\"\"
    name: str = Field(..., description="Name of the parameter.")
    type: str = Field(..., description="Python type hint as a string (e.g., 'str', 'int', 'float', 'List[float]', 'RasterData').")
    description: str = Field(..., description="Description of the parameter's purpose.")
    required: bool = Field(True, description="Whether the parameter is required.")
    default: Optional[Any] = Field(None, description="Default value if not provided.")
    enum: Optional[List[str]] = Field(None, description="List of allowed string values for enum types.")
    model_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for complex Pydantic model types.")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "input_raster",
                "type": "RasterData",
                "description": "The raster data to reproject.",
                "required": True,
                "model_schema": RasterData.model_json_schema()
            }
        }

class ToolDefinition(BaseModel):
    \"\"\"Defines a callable tool with its name, description, and parameters.\"\"\"
    name: str = Field(..., description="Unique name of the tool (e.g., 'reproject_raster_tool').")
    description: str = Field(..., description="A clear and concise description of what the tool does.")
    parameters: List[ToolParameter] = Field(..., description="List of parameters the tool accepts.")
    returns: str = Field(..., description="Description of what the tool returns (e.g., 'DataReprojectionResult', 'Dict[str, Any]').")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "reproject_raster_tool",
                "description": "Reprojects a raster dataset to a new Coordinate Reference System.",
                "parameters": [
                    {
                        "name": "input_raster_path",
                        "type": "str",
                        "description": "Path to the input raster file.",
                        "required": True
                    },
                    {
                        "name": "output_raster_path",
                        "type": "str",
                        "description": "Path for the reprojected output raster file.",
                        "required": True
                    },
                    {
                        "name": "target_crs",
                        "type": "str",
                        "description": "The target CRS (e.g., 'EPSG:4326').",
                        "required": True,
                        "enum": ["EPSG:4326", "EPSG:3857"]
                    },
                    {
                        "name": "resampling_method",
                        "type": "str",
                        "description": "Resampling algorithm.",
                        "required": False,
                        "default": "nearest",
                        "enum": ["nearest", "bilinear"]
                    }
                ],
                "returns": "DataReprojectionResult"
            }
        }

# --- Ontology for Planning ---

class Step(BaseModel):
    \"\"\"Represents a single step in a geospatial processing plan.\"\"\"
    tool_name: str = Field(..., description="The name of the tool to be executed in this step.")
    tool_args: Dict[str, Any] = Field(..., description="A dictionary of arguments for the tool, matching its ToolDefinition parameters.")
    output_key: str = Field(..., description="A unique key to store the output of this step in the overall plan results.")
    requires_input_from: Optional[str] = Field(None, description="Optional: The output_key of a previous step whose output is needed as input for this step.")

    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "download_bhoonidhi_data_tool",
                "tool_args": {
                    "aoi_geojson": {"type": "Polygon", "coordinates": [[[72,12],[72,15],[75,15],[75,12],[72,12]]]},
                    "dataset_type": "DEM",
                    "output_filename": "raw_dem.tif"
                },
                "output_key": "downloaded_dem"
            }
        }

class Plan(BaseModel):
    \"\"\"Represents a sequence of steps to achieve a geospatial processing goal.\"\"\"
    goal: str = Field(..., description="The overall goal of the plan.")
    steps: List[Step] = Field(..., description="An ordered list of steps to execute.")
    final_output_key: str = Field(..., description="The output_key of the step whose result represents the final output of the plan.")

    class Config:
        json_schema_extra = {
            "example": {
                "goal": "Download and reproject a DEM for a specific AOI.",
                "steps": [
                    {
                        "tool_name": "download_bhoonidhi_data_tool",
                        "tool_args": {
                            "aoi_geojson": {"type": "Polygon", "coordinates": [[[72,12],[72,15],[75,15],[75,12],[72,12]]]},
                            "dataset_type": "DEM",
                            "output_filename": "raw_dem.tif"
                        },
                        "output_key": "downloaded_dem"
                    },
                    {
                        "tool_name": "reproject_raster_tool",
                        "tool_args": {
                            "input_raster_path": "{downloaded_dem.path}", # Placeholder for dynamic input
                            "output_raster_path": "reprojected_dem.tif",
                            "target_crs": "EPSG:3857"
                        },
                        "output_key": "final_reprojected_dem",
                        "requires_input_from": "downloaded_dem"
                    }
                ],
                "final_output_key": "final_reprojected_dem"
            }
        }
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "ontology" / "ontology.py", ontology_content)

    # src/geospatial_agent/tool_manager/tool_registry.py
    tool_registry_content = """# src/geospatial_agent/tool_manager/tool_registry.py

from typing import Dict, Callable, Any, Optional, List
from loguru import logger

# Import the ToolDefinition ontology model
from geospatial_agent.ontology.ontology import ToolDefinition

class ToolRegistry:
    \"\"\"
    Manages a registry of available tools (functions) and their corresponding
    ToolDefinition metadata.

    This class serves as a central catalog for the geospatial agent, allowing it
    to discover and access tools based on their defined capabilities.
    \"\"\"

    def __init__(self):
        \"\"\"
        Initializes an empty ToolRegistry.
        \"\"\"
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(self, tool_definition: ToolDefinition, tool_function: Callable[..., Any]):
        \"\"\"
        Registers a tool function along with its metadata (ToolDefinition).

        Args:
            tool_definition (ToolDefinition): A Pydantic model defining the tool's capabilities.
            tool_function (Callable[..., Any]): The actual Python function implementing the tool.
        \"\"\"
        tool_name = tool_definition.name
        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' is already registered. Overwriting existing registration.")

        self._tools[tool_name] = tool_function
        self._tool_definitions[tool_name] = tool_definition
        logger.info(f"Tool '{tool_name}' registered successfully.")

    def get_tool(self, tool_name: str) -> Optional[Callable[..., Any]]:
        \"\"\"
        Retrieves a registered tool function by its name.

        Args:
            tool_name (str): The name of the tool to retrieve.

        Returns:
            Optional[Callable[..., Any]]: The tool function if found, otherwise None.
        \"\"\"
        tool_function = self._tools.get(tool_name)
        if tool_function is None:
            logger.warning(f"Attempted to retrieve unregistered tool: '{tool_name}'.")
        return tool_function

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        \"\"\"
        Retrieves a registered tool's definition by its name.

        Args:
            tool_name (str): The name of the tool's definition to retrieve.

        Returns:
            Optional[ToolDefinition]: The tool definition if found, otherwise None.
        \"\"\"
        tool_definition = self._tool_definitions.get(tool_name)
        if tool_definition is None:
            logger.warning(f"Attempted to retrieve definition for unregistered tool: '{tool_name}'.")
        return tool_definition

    def get_all_tool_definitions(self) -> List[ToolDefinition]:
        \"\"\"
        Retrieves a list of all registered tool definitions.

        Returns:
            List[ToolDefinition]: A list of all ToolDefinition objects.
        \"\"\"
        return list(self._tool_definitions.values())

    def clear_registry(self):
        \"\"\"
        Clears all registered tools and their definitions from the registry.
        Primarily for testing or re-initialization.
        \"\"\"
        self._tools.clear()
        self._tool_definitions.clear()
        logger.info("ToolRegistry cleared.")
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "tool_manager" / "tool_registry.py", tool_registry_content)

    # src/geospatial_agent/tool_manager/tool_manager.py
    tool_manager_content = """# src/geospatial_agent/tool_manager/tool_manager.py

from typing import Dict, Any, Callable, List, Optional, Union
from loguru import logger
from pydantic import ValidationError

# Import ToolRegistry
from geospatial_agent.tool_manager.tool_registry import ToolRegistry
# Import Ontology models for ToolDefinition and parameter schemas
from geospatial_agent.ontology.ontology import (
    ToolDefinition, ToolParameter,
    ReprojectRasterParams, ConvertVectorFormatParams, ClipRasterParams,
    CalculatePixelStatisticsParams, ReadVectorInfoParams, ReadRasterInfoParams, # Added ReadRasterInfoParams
    DownloadBhoonidhiDataParams, DownloadOSMDataParams,
    SpatialReferenceSystem, ResamplingMethod, VectorFormat,
    RasterData, VectorData # RasterData and VectorData are also in ontology
)
# Import concrete tool functions
from geospatial_agent.infrastructure.tools.gdal_tools import (
    reproject_raster_tool, convert_vector_format_tool, clip_raster_tool,
    calculate_pixel_statistics_tool, read_vector_info_tool, read_raster_info_tool, # Added read_raster_info_tool
    download_bhoonidhi_data_tool, download_osm_data_tool
)
# Import result schemas from domain.schemas
from geospatial_agent.domain.schemas import ( # Corrected import location
    DataReprojectionResult, DataConversionResult, DataDownloadResult
)


class ToolManager:
    \"\"\"
    Manages the registration and execution of geospatial tools.

    It initializes a ToolRegistry and populates it with definitions and
    functions for all available tools, including GDAL-based operations
    and data download functionalities.
    \"\"\"

    def __init__(self):
        \"\"\"
        Initializes the ToolManager, setting up the ToolRegistry and
        registering all known tools.
        \"\"\"
        self._registry = ToolRegistry()
        self._register_all_tools()
        logger.info("ToolManager initialized and tools registered.")

    def _register_all_tools(self):
        \"\"\"
        Registers all available geospatial tools with the ToolRegistry.
        Each tool is registered with its ToolDefinition (metadata) and
        its corresponding executable function.
        \"\"\"
        logger.info("Registering geospatial tools...")

        # 1. Reproject Raster Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="reproject_raster_tool",
                description="Reprojects a raster dataset to a new Coordinate Reference System (CRS).",
                parameters=[
                    ToolParameter(name="params", type="ReprojectRasterParams", description="Parameters for reprojection.", required=True, model_schema=ReprojectRasterParams.model_json_schema())
                ],
                returns="DataReprojectionResult"
            ),
            tool_function=reproject_raster_tool
        )

        # 2. Convert Vector Format Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="convert_vector_format_tool",
                description="Converts a vector dataset from one format to another (e.g., GeoJSON to Shapefile).",
                parameters=[
                    ToolParameter(name="params", type="ConvertVectorFormatParams", description="Parameters for vector conversion.", required=True, model_schema=ConvertVectorFormatParams.model_json_schema())
                ],
                returns="DataConversionResult"
            ),
            tool_function=convert_vector_format_tool
        )

        # 3. Clip Raster Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="clip_raster_tool",
                description="Clips a raster dataset using the geometries from a vector file.",
                parameters=[
                    ToolParameter(name="params", type="ClipRasterParams", description="Parameters for raster clipping.", required=True, model_schema=ClipRasterParams.model_json_schema())
                ],
                returns="RasterData" # Returns the metadata of the clipped raster
            ),
            tool_function=clip_raster_tool
        )

        # 4. Calculate Pixel Statistics Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="calculate_pixel_statistics_tool",
                description="Calculates basic pixel statistics (mean, standard deviation, min, max) for a specified band of a raster dataset.",
                parameters=[
                    ToolParameter(name="params", type="CalculatePixelStatisticsParams", description="Parameters for statistics calculation.", required=True, model_schema=CalculatePixelStatisticsParams.model_json_schema())
                ],
                returns="Dict[str, Any]" # Returns a dictionary of statistics
            ),
            tool_function=calculate_pixel_statistics_tool
        )

        # 5. Read Vector Info Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="read_vector_info_tool",
                description="Reads a vector file and extracts key metadata (CRS, layer name, geometry type, feature count).",
                parameters=[
                    ToolParameter(name="params", type="ReadVectorInfoParams", description="Parameters for reading vector info.", required=True, model_schema=ReadVectorInfoParams.model_json_schema())
                ],
                returns="VectorData" # Returns a Pydantic VectorData model
            ),
            tool_function=read_vector_info_tool
        )

        # 6. Read Raster Info Tool (NEW TOOL)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="read_raster_info_tool",
                description="Reads a raster file and extracts key metadata (CRS, dimensions, pixel size, band count, NoData value).",
                parameters=[
                    ToolParameter(name="params", type="ReadRasterInfoParams", description="Parameters for reading raster info.", required=True, model_schema=ReadRasterInfoParams.model_json_schema())
                ],
                returns="RasterData" # Returns a Pydantic RasterData model
            ),
            tool_function=read_raster_info_tool
        )

        # 7. Download Bhoonidhi Data Tool (Index changed from 6 to 7)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="download_bhoonidhi_data_tool",
                description="Downloads geospatial data (e.g., DEM, Satellite Imagery) from the Bhoonidhi platform for a specified Area of Interest (AOI).",
                parameters=[
                    ToolParameter(name="params", type="DownloadBhoonidhiDataParams", description="Parameters for Bhoonidhi download.", required=True, model_schema=DownloadBhoonidhiDataParams.model_json_schema())
                ],
                returns="DataDownloadResult"
            ),
            tool_function=download_bhoonidhi_data_tool
        )

        # 8. Download OSM Data Tool (Index changed from 7 to 8)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="download_osm_data_tool",
                description="Downloads OpenStreetMap (OSM) data (e.g., points of interest like schools, hospitals) using the Overpass API for a specified bounding box.",
                parameters=[
                    ToolParameter(name="params", type="DownloadOSMDataParams", description="Parameters for OSM data download.", required=True, model_schema=DownloadOSMDataParams.model_json_schema())
                ],
                returns="DataDownloadResult"
            ),
            tool_function=download_osm_data_tool
        )
        logger.info(f"Registered {len(self._registry.get_all_tool_definitions())} tools.")

    def get_tool_definitions(self) -> List[ToolDefinition]:
        \"\"\"
        Retrieves all registered tool definitions.

        Returns:
            List[ToolDefinition]: A list of ToolDefinition objects.
        \"\"\"
        return self._registry.get_all_tool_definitions()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Union[RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult, Dict[str, Any]]:
        \"\"\"
        Executes a registered tool with the given arguments.

        Args:
            tool_name (str): The name of the tool to execute.
            **kwargs (Any): Arguments to pass to the tool function. These arguments
                            are expected to conform to the tool's defined parameter schema.

        Returns:
            Union[RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult, Dict[str, Any]]:
                The result returned by the executed tool function. This will typically be
                a Pydantic model representing the operation's outcome or a dictionary for statistics.

        Raises:
            ValueError: If the tool is not found or if parameters do not match the schema.
        \"\"\"
        tool_function = self._registry.get_tool(tool_name)
        tool_definition = self._registry.get_tool_definition(tool_name)

        if not tool_function or not tool_definition:
            logger.error(f"Attempted to execute unregistered tool: '{{tool_name}}'.")
            raise ValueError(f"Tool '{{tool_name}}' not found in registry.")

        logger.info(f"Executing tool '{{tool_name}}' with arguments: {{kwargs}}")

        try:
            param_model_map = {
                "ReprojectRasterParams": ReprojectRasterParams,
                "ConvertVectorFormatParams": ConvertVectorFormatParams,
                "ClipRasterParams": ClipRasterParams,
                "CalculatePixelStatisticsParams": CalculatePixelStatisticsParams,
                "ReadVectorInfoParams": ReadVectorInfoParams,
                "ReadRasterInfoParams": ReadRasterInfoParams, # Added ReadRasterInfoParams
                "DownloadBhoonidhiDataParams": DownloadBhoonidhiDataParams,
                "DownloadOSMDataParams": DownloadOSMDataParams,
            }
            
            if tool_definition.parameters and tool_definition.parameters[0].name == "params":
                param_type_str = tool_definition.parameters[0].type
                param_model = param_model_map.get(param_type_str)
                if not param_model:
                    raise ValueError(f"Unknown parameter model type '{{param_type_str}}' for tool '{{tool_name}}'.")

                validated_params = param_model(**kwargs)
                result = tool_function(params=validated_params)
            else:
                result = tool_function(**kwargs)

            logger.success(f"Tool '{{tool_name}}' executed successfully.")
            return result
        except ValidationError as e:
            logger.error(f"Validation error for tool '{{tool_name}}' parameters: {{e.errors()}}")
            raise ValueError(f"Invalid parameters for tool '{{tool_name}}': {{e.errors()}}")
        except Exception as e:
            logger.error(f"Error executing tool '{{tool_name}}': {{e}}")
            raise RuntimeError(f"Tool execution failed for '{{tool_name}}': {{e}}")
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "tool_manager" / "tool_manager.py", tool_manager_content)

    # src/geospatial_agent/application/agents/planner_agent.py
    planner_agent_content = """# src/geospatial_agent/application/agents/planner_agent.py

from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
import json
import os

from geospatial_agent.tool_manager.tool_manager import ToolManager
from geospatial_agent.ontology.ontology import Plan, ToolDefinition, Step
from geospatial_agent.ontology.ontology import (
    ReprojectRasterParams, ConvertVectorFormatParams, ClipRasterParams,
    CalculatePixelStatisticsParams, ReadVectorInfoParams, ReadRasterInfoParams, # Ensure ReadRasterInfoParams is imported
    DownloadBhoonidhiDataParams, DownloadOSMDataParams,
    SpatialReferenceSystem, ResamplingMethod, VectorFormat,
    RasterData, VectorData
)

class PlannerAgent:
    \"\"\"
    The PlannerAgent is responsible for generating a structured execution plan
    (a sequence of tool calls) based on a natural language user request.
    \"\"\"

    def __init__(self, tool_manager: ToolManager, openai_api_key: Optional[str] = None):
        \"\"\"
        Initializes the PlannerAgent.
        \"\"\"
        self.tool_manager = tool_manager
        self.client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = "gpt-4o-mini"
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")

    def _get_tool_schemas_for_llm(self) -> List[Dict[str, Any]]:
        \"\"\"
        Retrieves the JSON schemas for all registered tools, formatted for the LLM.
        \"\"\"
        tool_definitions = self.tool_manager.get_tool_definitions() # This now correctly gets all 8 tools
        llm_tools = []
        for td in tool_definitions:
            if td.parameters and td.parameters[0].name == "params" and td.parameters[0].model_schema:
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters[0].model_schema
                    }
                })
            else:
                logger.warning(f"Tool '{td.name}' does not have a valid 'params' model schema defined for LLM use.")
        return llm_tools

    def generate_plan(self, user_query: str) -> Plan:
        \"\"\"
        Generates a geospatial processing plan based on the user's query.
        \"\"\"
        logger.info(f"Generating plan for query: '{user_query}'")

        available_tools_for_llm = self._get_tool_schemas_for_llm()
        if not available_tools_for_llm:
            logger.error("No valid tool schemas available for the LLM to use.")
            raise ValueError("No geospatial tools are registered or their definitions are invalid.")

        plan_tool_schema = {
            "type": "function",
            "function": {
                "name": "create_geospatial_plan",
                "description": "Creates a sequence of steps (a plan) using available geospatial tools to achieve a user's goal.",
                "parameters": Plan.model_json_schema()
            }
        }

        messages = [
            {
                "role": "system",
                "content": f\"\"\"You are a highly intelligent geospatial planning agent.
            Your primary goal is to generate a structured execution plan (a 'Plan' object)
            to fulfill a user's geospatial data processing request.
            
            You have access to a set of specialized geospatial tools.
            You MUST use the provided tool definitions to construct a JSON plan.
            
            When a user asks for a geospatial task, you should respond by calling the `create_geospatial_plan` function.
            The `create_geospatial_plan` function expects a 'Plan' object as its argument.
            
            The 'Plan' object has the following structure:
            ```json
            {{
                "goal": "...",
                "steps": [
                    {{
                        "tool_name": "...",
                        "tool_args": {{
                            // Arguments for the tool go here
                        }},
                        "output_key": "...",
                        "requires_input_from": "..." // Optional
                    }}
                ],
                "final_output_key": "..."
            }}
            ```
            
            Here are the available geospatial tools and their schemas:
            {json.dumps(available_tools_for_llm, indent=2)}
            
            Carefully consider the user's request and the capabilities of the tools.
            
            **CRITICAL INSTRUCTION FOR TOOL ARGUMENTS:**
            
            All arguments for a tool MUST be placed inside the `tool_args` dictionary within each `Step`.
            DO NOT use any other key (like 'parameters') for tool arguments.
            
            If a tool's parameter is a Pydantic model (like `RasterData` or `VectorData`), you MUST provide its fields as a DICTIONARY within `tool_args`.
            
            **Example 1: Calling a tool with a direct file path for a nested RasterData parameter:**
            ```json
            {{
                "tool_name": "reproject_raster_tool",
                "tool_args": {{
                    "input_raster": {{"path": "/data/my_image.tif", "data_type": "raster", "crs": "EPSG:4326"}},
                    "target_crs": "EPSG:3857",
                    "output_path": "/data/reprojected_image.tif",
                    "resampling_method": "bilinear"
                }},
                "output_key": "reprojected_image"
            }}
            ```
            
            **Example 2: Using a placeholder for a nested RasterData parameter where the path comes from a previous step's output:**
            ```json
            {{
                "tool_name": "reproject_raster_tool",
                "tool_args": {{
                    "input_raster": {{"path": "{{downloaded_dem.path}}", "data_type": "raster", "crs": "EPSG:4326"}},
                    "target_crs": "EPSG:3857",
                    "output_path": "reprojected_dem.tif",
                    "resampling_method": "nearest"
                }},
                "output_key": "final_reprojected_dem",
                "requires_input_from": "downloaded_dem"
            }}
            ```
            
            Always ensure 'data_type' for `RasterData` is explicitly set to 'raster' and for `VectorData` to 'vector'.
            
            Your response MUST be a single call to `create_geospatial_plan`.
            Do not include any conversational text outside of the tool call.
            \"\"\"
            },
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[plan_tool_schema],
                tool_choice={"type": "function", "function": {"name": "create_geospatial_plan"}},
            )
            
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "create_geospatial_plan":
                    plan_args = json.loads(tool_call.function.arguments)
                    plan = Plan(**plan_args)
                    logger.success("Successfully generated geospatial plan from LLM.")
                    return plan
                else:
                    raise ValueError(f"LLM called unexpected tool: {{tool_call.function.name}}")
            else:
                raise ValueError("LLM did not return a tool call for plan generation.")

        except ValidationError as e:
            logger.error(f"Pydantic validation error when creating plan: {{e.errors()}}")
            raise ValueError(f"LLM generated an invalid plan: {{e.errors()}}")
        except Exception as e:
            logger.error(f"Error generating plan with LLM: {{e}}")
            raise RuntimeError(f"Failed to generate plan: {{e}}")
"""
    write_file_content(GEOSPATIAL_AGENT_DIR / "application" / "agents" / "planner_agent.py", planner_agent_content)


    print("\n--- 3. Creating dummy geospatial data for testing ---")
    data_downloads_dir = PROJECT_ROOT / "data" / "downloads"
    tool_outputs_dir = PROJECT_ROOT / "data" / "tool_outputs"
    ensure_directory(data_downloads_dir)
    ensure_directory(tool_outputs_dir)
    print(f"Created data directories: {data_downloads_dir} and {tool_outputs_dir}")

    create_dummy_geotiff(data_downloads_dir / "dummy_dem.tif")
    create_dummy_geojson(data_downloads_dir / "dummy_aoi.geojson")

    print("\n--- Automated Project Setup Complete ---")


if __name__ == "__main__":
    setup_project()