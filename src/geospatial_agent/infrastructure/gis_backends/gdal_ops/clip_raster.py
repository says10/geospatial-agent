# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/clip_raster.py

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
    """
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
    """
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
            # `crop=True` crops the output raster to the extent of the geometries
            # `nodata=src.nodata` ensures the NoData value is preserved in the output
            out_image, out_transform = mask(src, geometries, crop=True, nodata=src.nodata)
            logger.debug(f"Raster masked. Output image shape: {out_image.shape}")

            # Update metadata for the clipped raster
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff", # Ensure output is GeoTIFF
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
            width=0, height=0, pixel_size_x=0.0, pixel_size_y=0.
