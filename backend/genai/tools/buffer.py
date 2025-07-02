import geopandas as gpd
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def buffer_layer(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    distance: float
) -> Union[Path, None]:
    """
    Creates a buffer zone around features in a geospatial layer.

    It checks for projected vs. geographic CRS to warn about ambiguous distance units.

    Args:
        input_path: Path to the input geospatial file.
        output_path: Path where the buffered file will be saved.
        distance: The buffer distance. The unit is derived from the layer's CRS
                  (e.g., meters for projected, degrees for geographic).

    Returns:
        The path to the output file on success, or None on failure.
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        logger.info(f"Reading layer: {input_path}")
        gdf = gpd.read_file(input_path)

        if gdf.crs.is_geographic:
            logger.warning(
                f"The layer's CRS ({gdf.crs.name}) is geographic. "
                f"The buffer distance of {distance} will be interpreted in degrees, "
                "which may lead to unexpected results. It is highly recommended to "
                "reproject to a projected CRS (e.g., a local UTM zone) first."
            )

        logger.info(f"Creating buffer of {distance} units...")
        
        gdf.geometry = gdf.geometry.buffer(distance)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path)
        logger.info(f"Buffer created successfully. Result saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"An error occurred during buffering: {e}", exc_info=True)
        return None