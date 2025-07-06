# src/geospatial_agent/data_ingestion/data_loader.py

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
    """
    Manages the loading, downloading, and basic processing of geospatial data.

    This class acts as an interface between the core agent logic and the underlying
    geospatial data infrastructure. It can simulate data downloads and orchestrate
    calls to specific GDAL-based operations.
    """

    def __init__(self, data_storage_dir: Union[str, Path] = "data/tool_outputs"):
        """
        Initializes the DataLoader.

        Args:
            data_storage_dir (Union[str, Path]): The base directory where downloaded
                                                  and processed data will be stored.
        """
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
        """
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
        """
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
        """
        Simulates downloading OpenStreetMap (OSM) data using the Overpass API.

        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat].
            amenity_type: Type of amenity to query (e.g., "school", "hospital").
            output_filename: Desired filename for the downloaded data (e.g., "schools.geojson").
            overpass_url: URL for the Overpass API endpoint.

        Returns:
            DataDownloadResult: A Pydantic model representing the result of the download.
        """
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
        """
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
        """
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
