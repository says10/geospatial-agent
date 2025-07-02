import geopandas as gpd
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def intersect_layers(
    layer1_path: Union[str, Path],
    layer2_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Union[Path, None]:
    """
    Computes the geometric intersection of two geospatial layers.

    This function identifies the overlapping regions between two vector layers and
    saves the result. It handles CRS reprojection and ensures inputs are valid.

    Args:
        layer1_path: Path to the first input geospatial file.
        layer2_path: Path to the second input geospatial file.
        output_path: Path where the resulting intersection file will be saved.

    Returns:
        The path to the output file on success, or None on failure.
    """
    try:
        layer1_path = Path(layer1_path)
        layer2_path = Path(layer2_path)
        output_path = Path(output_path)

        if not layer1_path.exists():
            logger.error(f"Input file not found: {layer1_path}")
            return None
        if not layer2_path.exists():
            logger.error(f"Input file not found: {layer2_path}")
            return None

        logger.info(f"Reading layers: {layer1_path} and {layer2_path}")
        gdf1 = gpd.read_file(layer1_path)
        gdf2 = gpd.read_file(layer2_path)

        if gdf1.crs != gdf2.crs:
            logger.warning(f"CRS mismatch detected. Reprojecting {layer2_path.name} to match {layer1_path.name}'s CRS ({gdf1.crs}).")
            gdf2 = gdf2.to_crs(gdf1.crs)

        logger.info("Performing intersection...")
        intersection_gdf = gpd.overlay(gdf1, gdf2, how='intersection')

        if intersection_gdf.empty:
            logger.warning("The layers do not intersect. No output file will be created.")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        intersection_gdf.to_file(output_path)
        logger.info(f"Intersection successful. Result saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"An error occurred during intersection: {e}", exc_info=True)
        return None