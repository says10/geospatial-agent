# src/geospatial_agent/infrastructure/gis_backends/gdal_ops/read_shapefile.py

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
    """
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
    """
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
