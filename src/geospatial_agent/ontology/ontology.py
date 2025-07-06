# src/geospatial_agent/ontology/ontology.py

from pydantic import BaseModel, Field, conlist, confloat
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum

# --- Enums for common geospatial concepts ---

class SpatialReferenceSystem(str, Enum):
    """Commonly used Spatial Reference Systems (CRS)."""
    EPSG_4326 = "EPSG:4326" # WGS 84 Geographic (Lat/Lon)
    EPSG_3857 = "EPSG:3857" # Web Mercator
    EPSG_32644 = "EPSG:32644" # WGS 84 / UTM zone 44N (example for India)
    # Add more as needed

class RasterFormat(str, Enum):
    """Supported Raster Formats."""
    GEOTIFF = "GTiff"
    PNG = "PNG"
    JPEG = "JPEG"
    # Add more as needed

class VectorFormat(str, Enum):
    """Supported Vector Formats."""
    GEOJSON = "GeoJSON"
    SHAPEFILE = "ESRI Shapefile"
    GPKG = "GPKG" # GeoPackage
    FLATGEBUF = "FlatGeobuf"
    # Add more as needed

class ResamplingMethod(str, Enum):
    """Resampling methods for raster operations."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    # Add more as needed

class StatisticType(str, Enum):
    """Types of statistics that can be calculated."""
    MEAN = "mean"
    STD_DEV = "std_dev"
    MIN = "min"
    MAX = "max"
    # Add more as needed

# --- Core Ontology Models ---

class GeospatialDataType(str, Enum):
    """Defines the high-level type of geospatial data."""
    RASTER = "raster"
    VECTOR = "vector"

class SpatialExtent(BaseModel):
    """Represents a geographic bounding box."""
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
    """Abstract base model for any geospatial data."""
    path: str = Field(..., description="File path to the geospatial data.")
    data_type: GeospatialDataType = Field(..., description="High-level type of the geospatial data (raster or vector).")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g., 'EPSG:4326').")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as key-value pairs.")

class RasterData(GeospatialData):
    """Represents a raster dataset."""
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
    """Represents a vector dataset."""
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
    """Parameters for raster reprojection."""
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
    """Parameters for vector format conversion."""
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
    """Parameters for clipping a raster with a vector."""
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
    """Parameters for calculating raster pixel statistics."""
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
    """Parameters for reading vector file information."""
    input_vector: str = Field(..., description="Path to the input vector file.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_vector": "/data/countries.shp"
            }
        }

class ReadRasterInfoParams(BaseModel):
    """Parameters for reading raster file information."""
    input_raster: str = Field(..., description="Path to the input raster file.")

    class Config:
        json_schema_extra = {
            "example": {
                "input_raster": "/data/dem.tif"
            }
        }

class DownloadBhoonidhiDataParams(BaseModel):
    """Parameters for downloading data from Bhoonidhi."""
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
    """Parameters for downloading OpenStreetMap data via Overpass API."""
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
    """Represents a single parameter for a tool."""
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
    """Defines a callable tool with its name, description, and parameters."""
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
    """Represents a single step in a geospatial processing plan."""
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
    """Represents a sequence of steps to achieve a geospatial processing goal."""
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
