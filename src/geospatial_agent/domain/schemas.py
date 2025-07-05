# src/geospatial_agent/domain/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RasterData(BaseModel):
    """
    Represents raster data with essential metadata.

    Attributes:
        path (str): File path to the raster data.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326', 'ESRI:102008').
        width (int): Width of the raster in pixels.
        height (int): Height of the raster in pixels.
        pixel_size_x (float): Pixel size in the X direction (georeferenced units).
        pixel_size_y (float): Pixel size in the Y direction (georeferenced units).
        band_count (int): Number of bands in the raster.
        no_data_value (Optional[float]): NoData value if present in the raster. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata as a dictionary. Defaults to None.
    """
    path: str = Field(..., description="File path to the raster data.")
    crs: str = Field(..., description="Coordinate Reference System (e.g., 'EPSG:4326').")
    width: int = Field(..., description="Width of the raster in pixels.")
    height: int = Field(..., description="Height of the raster in pixels.")
    pixel_size_x: float = Field(..., description="Pixel size in X direction (georeferenced units).")
    pixel_size_y: float = Field(..., description="Pixel size in Y direction (georeferenced units).")
    band_count: int = Field(..., description="Number of bands in the raster.")
    no_data_value: Optional[float] = Field(None, description="NoData value if present.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as a dictionary.")

class VectorData(BaseModel):
    """
    Represents vector data with essential metadata.

    Attributes:
        path (str): File path to the vector data.
        crs (str): Coordinate Reference System (e.g., 'EPSG:4326').
        layer_name (str): Name of the primary layer within the vector file.
        geometry_type (Optional[str]): Type of geometry (e.g., 'Polygon', 'Point', 'LineString').
                                       Defaults to None.
        feature_count (Optional[int]): Number of features in the vector layer. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata as a dictionary. Defaults to None.
    """
    path: str = Field(..., description="File path to the vector data.")
    crs: str = Field(..., description="Coordinate Reference System (e.g., 'EPSG:4326').")
    layer_name: str = Field(..., description="Name of the primary layer within the vector file.")
    geometry_type: Optional[str] = Field(None, description="Type of geometry (e.g., 'Polygon', 'Point', 'LineString').")
    feature_count: Optional[int] = Field(None, description="Number of features in the vector layer.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata as a dictionary.")

class AreaCalculationResult(BaseModel):
    """
    Represents the result of an area calculation, potentially with slope correction.

    Attributes:
        area_sq_m (float): Calculated area in square meters.
        unit (str): Unit of the calculated area. Defaults to "square meters".
        notes (Optional[str]): Any additional notes about the calculation (e.g., 'slope corrected').
                               Defaults to None.
    """
    area_sq_m: float = Field(..., description="Calculated area in square meters.")
    unit: str = Field("square meters", description="Unit of the calculated area.")
    notes: Optional[str] = Field(None, description="Any additional notes about the calculation (e.g., 'slope corrected').")

class DataDownloadResult(BaseModel):
    """
    Represents the result of downloading data.

    Attributes:
        path (str): Local file path to the downloaded data.
        data_source (str): Source from which the data was downloaded (e.g., 'Bhoonidhi', 'OpenStreetMap').
        original_url (Optional[str]): Original URL if downloaded from a web source. Defaults to None.
        file_type (str): Type of the downloaded file (e.g., 'GeoTIFF', 'Shapefile', 'GeoJSON').
        description (Optional[str]): A brief description of the downloaded data. Defaults to None.
    """
    path: str = Field(..., description="Local file path to the downloaded data.")
    data_source: str = Field(..., description="Source from which the data was downloaded (e.g., 'Bhoonidhi', 'OpenStreetMap').")
    original_url: Optional[str] = Field(None, description="Original URL if downloaded from a web source.")
    file_type: str = Field(..., description="Type of the downloaded file (e.g., 'GeoTIFF', 'Shapefile', 'GeoJSON').")
    description: Optional[str] = Field(None, description="A brief description of the downloaded data.")

class DataConversionResult(BaseModel):
    """
    Represents the result of converting data from one format to another.

    Attributes:
        original_path (str): Path to the original data file.
        converted_path (str): Path to the newly converted data file.
        original_format (str): Original format of the data (e.g., 'Shapefile', 'GeoTIFF').
        new_format (str): New format of the data (e.g., 'GeoJSON', 'COG').
        success (bool): True if conversion was successful, False otherwise.
        message (Optional[str]): Details about the conversion process or any errors. Defaults to None.
    """
    original_path: str = Field(..., description="Path to the original data file.")
    converted_path: str = Field(..., description="Path to the newly converted data file.")
    original_format: str = Field(..., description="Original format of the data (e.g., 'Shapefile', 'GeoTIFF').")
    new_format: str = Field(..., description="New format of the data (e.g., 'GeoJSON', 'COG').")
    success: bool = Field(..., description="True if conversion was successful, False otherwise.")
    message: Optional[str] = Field(None, description="Details about the conversion process or any errors.")

class DataReprojectionResult(BaseModel):
    """
    Represents the result of reprojecting geospatial data.

    Attributes:
        original_path (str): Path to the original data file.
        reprojected_path (str): Path to the newly reprojected data file.
        original_crs (str): Original Coordinate Reference System (e.g., 'EPSG:4326').
        new_crs (str): New Coordinate Reference System (e.g., 'EPSG:3857').
        success (bool): True if reprojection was successful, False otherwise.
        message (Optional[str]): Details about the reprojection process or any errors. Defaults to None.
    """
    original_path: str = Field(..., description="Path to the original data file.")
    reprojected_path: str = Field(..., description="Path to the newly reprojected data file.")
    original_crs: str = Field(..., description="Original Coordinate Reference System (e.g., 'EPSG:4326').")
    new_crs: str = Field(..., description="New Coordinate Reference System (e.g., 'EPSG:3857').")
    success: bool = Field(..., description="True if reprojection was successful, False otherwise.")
    message: Optional[str] = Field(None, description="Details about the reprojection process or any errors.")
