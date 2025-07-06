# src/geospatial_agent/tool_manager/tool_manager.py

from typing import Dict, Any, Callable, List, Optional, Union
from loguru import logger
from pydantic import ValidationError

# Import ToolRegistry
from geospatial_agent.tool_manager.tool_registry import ToolRegistry
# Import Ontology models for ToolDefinition and parameter schemas
from geospatial_agent.ontology.ontology import (
    ToolDefinition, ToolParameter,
    ReprojectRasterParams, ConvertVectorFormatParams, ClipRasterParams,
    CalculatePixelStatisticsParams, ReadVectorInfoParams, ReadRasterInfoParams, # Added ReadRasterInfoParams
    DownloadBhoonidhiDataParams, DownloadOSMDataParams,
    SpatialReferenceSystem, ResamplingMethod, VectorFormat,
    RasterData, VectorData # RasterData and VectorData are also in ontology
)
# Import concrete tool functions
from geospatial_agent.infrastructure.tools.gdal_tools import (
    reproject_raster_tool, convert_vector_format_tool, clip_raster_tool,
    calculate_pixel_statistics_tool, read_vector_info_tool, read_raster_info_tool, # Added read_raster_info_tool
    download_bhoonidhi_data_tool, download_osm_data_tool
)
# Import result schemas from domain.schemas
from geospatial_agent.domain.schemas import ( # Corrected import location
    DataReprojectionResult, DataConversionResult, DataDownloadResult
)


class ToolManager:
    """
    Manages the registration and execution of geospatial tools.

    It initializes a ToolRegistry and populates it with definitions and
    functions for all available tools, including GDAL-based operations
    and data download functionalities.
    """

    def __init__(self):
        """
        Initializes the ToolManager, setting up the ToolRegistry and
        registering all known tools.
        """
        self._registry = ToolRegistry()
        self._register_all_tools()
        logger.info("ToolManager initialized and tools registered.")

    def _register_all_tools(self):
        """
        Registers all available geospatial tools with the ToolRegistry.
        Each tool is registered with its ToolDefinition (metadata) and
        its corresponding executable function.
        """
        logger.info("Registering geospatial tools...")

        # 1. Reproject Raster Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="reproject_raster_tool",
                description="Reprojects a raster dataset to a new Coordinate Reference System (CRS).",
                parameters=[
                    ToolParameter(name="params", type="ReprojectRasterParams", description="Parameters for reprojection.", required=True, model_schema=ReprojectRasterParams.model_json_schema())
                ],
                returns="DataReprojectionResult"
            ),
            tool_function=reproject_raster_tool
        )

        # 2. Convert Vector Format Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="convert_vector_format_tool",
                description="Converts a vector dataset from one format to another (e.g., GeoJSON to Shapefile).",
                parameters=[
                    ToolParameter(name="params", type="ConvertVectorFormatParams", description="Parameters for vector conversion.", required=True, model_schema=ConvertVectorFormatParams.model_json_schema())
                ],
                returns="DataConversionResult"
            ),
            tool_function=convert_vector_format_tool
        )

        # 3. Clip Raster Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="clip_raster_tool",
                description="Clips a raster dataset using the geometries from a vector file.",
                parameters=[
                    ToolParameter(name="params", type="ClipRasterParams", description="Parameters for raster clipping.", required=True, model_schema=ClipRasterParams.model_json_schema())
                ],
                returns="RasterData" # Returns the metadata of the clipped raster
            ),
            tool_function=clip_raster_tool
        )

        # 4. Calculate Pixel Statistics Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="calculate_pixel_statistics_tool",
                description="Calculates basic pixel statistics (mean, standard deviation, min, max) for a specified band of a raster dataset.",
                parameters=[
                    ToolParameter(name="params", type="CalculatePixelStatisticsParams", description="Parameters for statistics calculation.", required=True, model_schema=CalculatePixelStatisticsParams.model_json_schema())
                ],
                returns="Dict[str, Any]" # Returns a dictionary of statistics
            ),
            tool_function=calculate_pixel_statistics_tool
        )

        # 5. Read Vector Info Tool
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="read_vector_info_tool",
                description="Reads a vector file and extracts key metadata (CRS, layer name, geometry type, feature count).",
                parameters=[
                    ToolParameter(name="params", type="ReadVectorInfoParams", description="Parameters for reading vector info.", required=True, model_schema=ReadVectorInfoParams.model_json_schema())
                ],
                returns="VectorData" # Returns a Pydantic VectorData model
            ),
            tool_function=read_vector_info_tool
        )

        # 6. Read Raster Info Tool (NEW TOOL)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="read_raster_info_tool",
                description="Reads a raster file and extracts key metadata (CRS, dimensions, pixel size, band count, NoData value).",
                parameters=[
                    ToolParameter(name="params", type="ReadRasterInfoParams", description="Parameters for reading raster info.", required=True, model_schema=ReadRasterInfoParams.model_json_schema())
                ],
                returns="RasterData" # Returns a Pydantic RasterData model
            ),
            tool_function=read_raster_info_tool
        )

        # 7. Download Bhoonidhi Data Tool (Index changed from 6 to 7)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="download_bhoonidhi_data_tool",
                description="Downloads geospatial data (e.g., DEM, Satellite Imagery) from the Bhoonidhi platform for a specified Area of Interest (AOI).",
                parameters=[
                    ToolParameter(name="params", type="DownloadBhoonidhiDataParams", description="Parameters for Bhoonidhi download.", required=True, model_schema=DownloadBhoonidhiDataParams.model_json_schema())
                ],
                returns="DataDownloadResult"
            ),
            tool_function=download_bhoonidhi_data_tool
        )

        # 8. Download OSM Data Tool (Index changed from 7 to 8)
        self._registry.register_tool(
            tool_definition=ToolDefinition(
                name="download_osm_data_tool",
                description="Downloads OpenStreetMap (OSM) data (e.g., points of interest like schools, hospitals) using the Overpass API for a specified bounding box.",
                parameters=[
                    ToolParameter(name="params", type="DownloadOSMDataParams", description="Parameters for OSM data download.", required=True, model_schema=DownloadOSMDataParams.model_json_schema())
                ],
                returns="DataDownloadResult"
            ),
            tool_function=download_osm_data_tool
        )
        logger.info(f"Registered {len(self._registry.get_all_tool_definitions())} tools.")

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Retrieves all registered tool definitions.

        Returns:
            List[ToolDefinition]: A list of ToolDefinition objects.
        """
        return self._registry.get_all_tool_definitions()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Union[RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult, Dict[str, Any]]:
        """
        Executes a registered tool with the given arguments.

        Args:
            tool_name (str): The name of the tool to execute.
            **kwargs (Any): Arguments to pass to the tool function. These arguments
                            are expected to conform to the tool's defined parameter schema.

        Returns:
            Union[RasterData, VectorData, DataReprojectionResult, DataConversionResult, DataDownloadResult, Dict[str, Any]]:
                The result returned by the executed tool function. This will typically be
                a Pydantic model representing the operation's outcome or a dictionary for statistics.

        Raises:
            ValueError: If the tool is not found or if parameters do not match the schema.
        """
        tool_function = self._registry.get_tool(tool_name)
        tool_definition = self._registry.get_tool_definition(tool_name)

        if not tool_function or not tool_definition:
            logger.error(f"Attempted to execute unregistered tool: '{{tool_name}}'.")
            raise ValueError(f"Tool '{{tool_name}}' not found in registry.")

        logger.info(f"Executing tool '{{tool_name}}' with arguments: {{kwargs}}")

        try:
            param_model_map = {
                "ReprojectRasterParams": ReprojectRasterParams,
                "ConvertVectorFormatParams": ConvertVectorFormatParams,
                "ClipRasterParams": ClipRasterParams,
                "CalculatePixelStatisticsParams": CalculatePixelStatisticsParams,
                "ReadVectorInfoParams": ReadVectorInfoParams,
                "ReadRasterInfoParams": ReadRasterInfoParams, # Added ReadRasterInfoParams
                "DownloadBhoonidhiDataParams": DownloadBhoonidhiDataParams,
                "DownloadOSMDataParams": DownloadOSMDataParams,
            }
            
            if tool_definition.parameters and tool_definition.parameters[0].name == "params":
                param_type_str = tool_definition.parameters[0].type
                param_model = param_model_map.get(param_type_str)
                if not param_model:
                    raise ValueError(f"Unknown parameter model type '{{param_type_str}}' for tool '{{tool_name}}'.")

                validated_params = param_model(**kwargs)
                result = tool_function(params=validated_params)
            else:
                result = tool_function(**kwargs)

            logger.success(f"Tool '{{tool_name}}' executed successfully.")
            return result
        except ValidationError as e:
            logger.error(f"Validation error for tool '{{tool_name}}' parameters: {{e.errors()}}")
            raise ValueError(f"Invalid parameters for tool '{{tool_name}}': {{e.errors()}}")
        except Exception as e:
            logger.error(f"Error executing tool '{{tool_name}}': {{e}}")
            raise RuntimeError(f"Tool execution failed for '{{tool_name}}': {{e}}")
