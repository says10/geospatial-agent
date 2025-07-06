# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/calculate_pixel_statistics.py

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
    """
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
    """
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
