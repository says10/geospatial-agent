# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/reproject_raster.py

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
from typing import Union
from loguru import logger # For logging errors and important events
import os # For os.cpu_count()

# Assuming geospatial_agent is installed or reachable in sys.path
from geospatial_agent.domain.schemas import DataReprojectionResult

@logger.catch(reraise=False) # Decorator to log exceptions without re-raising by default
def reproject_raster(
    input_raster_path: Union[str, Path],
    output_raster_path: Union[str, Path],
    target_crs: str,
    resampling_method: Resampling = Resampling.nearest,
) -> DataReprojectionResult:
    """
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
    """
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
