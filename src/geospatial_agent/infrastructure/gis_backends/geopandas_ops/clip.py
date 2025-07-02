import geopandas as gpd
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def clip_layer(
    input_path: Union[str, Path],
    clip_layer_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Union[Path, None]:
    """
    Clips an input layer to the extent of a clip layer.

    This function acts like a cookie-cutter, keeping only the parts of the
    input_layer that fall within the boundaries of the clip_layer.

    Args:
        input_path: Path to the geospatial file to be clipped.
        clip_layer_path: Path to the geospatial file used for clipping.
        output_path: Path where the clipped file will be saved.

    Returns:
        The path to the output file on success, or None on failure.
    """
    try:
        input_path = Path(input_path)
        clip_layer_path = Path(clip_layer_path)
        output_path = Path(output_path)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None
        if not clip_layer_path.exists():
            logger.error(f"Clip layer file not found: {clip_layer_path}")
            return None

        logger.info(f"Reading input layer {input_path.name} and clip layer {clip_layer_path.name}")
        input_gdf = gpd.read_file(input_path)
        clip_gdf = gpd.read_file(clip_layer_path)

        if input_gdf.crs != clip_gdf.crs:
            logger.warning(f"CRS mismatch detected. Reprojecting clip layer to match input layer's CRS ({input_gdf.crs}).")
            clip_gdf = clip_gdf.to_crs(input_gdf.crs)

        logger.info("Performing clip operation...")
        clipped_gdf = gpd.clip(input_gdf, clip_gdf)

        if clipped_gdf.empty:
            logger.warning("The clip operation resulted in an empty layer. No output file will be created.")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        clipped_gdf.to_file(output_path)
        logger.info(f"Clip successful. Result saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"An error occurred during clip: {e}", exc_info=True)
        return None