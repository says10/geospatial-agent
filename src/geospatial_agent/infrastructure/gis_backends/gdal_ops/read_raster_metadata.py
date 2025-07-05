# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_raster_metadata.py

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
    """
    Reads a raster file and extracts key metadata (CRS, dimensions, pixel size, band count).

    Args:
        input_raster_path (Union[str, Path]): Path to the input raster file.

    Returns:
        RasterData: A Pydantic model containing extracted metadata about the raster file.
    """
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
