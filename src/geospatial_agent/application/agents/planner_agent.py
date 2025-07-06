# src/geospatial_agent/application/agents/planner_agent.py

from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
import json
import os

from geospatial_agent.tool_manager.tool_manager import ToolManager
from geospatial_agent.ontology.ontology import Plan, ToolDefinition, Step
from geospatial_agent.ontology.ontology import (
    ReprojectRasterParams, ConvertVectorFormatParams, ClipRasterParams,
    CalculatePixelStatisticsParams, ReadVectorInfoParams, ReadRasterInfoParams, # Ensure ReadRasterInfoParams is imported
    DownloadBhoonidhiDataParams, DownloadOSMDataParams,
    SpatialReferenceSystem, ResamplingMethod, VectorFormat,
    RasterData, VectorData
)

class PlannerAgent:
    """
    The PlannerAgent is responsible for generating a structured execution plan
    (a sequence of tool calls) based on a natural language user request.
    """

    def __init__(self, tool_manager: ToolManager, openai_api_key: Optional[str] = None):
        """
        Initializes the PlannerAgent.
        """
        self.tool_manager = tool_manager
        self.client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = "gpt-4o-mini"
        logger.info(f"PlannerAgent initialized with model: {self.model_name}")

    def _get_tool_schemas_for_llm(self) -> List[Dict[str, Any]]:
        """
        Retrieves the JSON schemas for all registered tools, formatted for the LLM.
        """
        tool_definitions = self.tool_manager.get_tool_definitions() # This now correctly gets all 8 tools
        llm_tools = []
        for td in tool_definitions:
            if td.parameters and td.parameters[0].name == "params" and td.parameters[0].model_schema:
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": td.name,
                        "description": td.description,
                        "parameters": td.parameters[0].model_schema
                    }
                })
            else:
                logger.warning(f"Tool '{td.name}' does not have a valid 'params' model schema defined for LLM use.")
        return llm_tools

    def generate_plan(self, user_query: str) -> Plan:
        """
        Generates a geospatial processing plan based on the user's query.
        """
        logger.info(f"Generating plan for query: '{user_query}'")

        available_tools_for_llm = self._get_tool_schemas_for_llm()
        if not available_tools_for_llm:
            logger.error("No valid tool schemas available for the LLM to use.")
            raise ValueError("No geospatial tools are registered or their definitions are invalid.")

        plan_tool_schema = {
            "type": "function",
            "function": {
                "name": "create_geospatial_plan",
                "description": "Creates a sequence of steps (a plan) using available geospatial tools to achieve a user's goal.",
                "parameters": Plan.model_json_schema()
            }
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a highly intelligent geospatial planning agent.
            Your primary goal is to generate a structured execution plan (a 'Plan' object)
            to fulfill a user's geospatial data processing request.
            
            You have access to a set of specialized geospatial tools.
            You MUST use the provided tool definitions to construct a JSON plan.
            
            When a user asks for a geospatial task, you should respond by calling the `create_geospatial_plan` function.
            The `create_geospatial_plan` function expects a 'Plan' object as its argument.
            
            The 'Plan' object has the following structure:
            ```json
            {{
                "goal": "...",
                "steps": [
                    {{
                        "tool_name": "...",
                        "tool_args": {{
                            // Arguments for the tool go here
                        }},
                        "output_key": "...",
                        "requires_input_from": "..." // Optional
                    }}
                ],
                "final_output_key": "..."
            }}
            ```
            
            Here are the available geospatial tools and their schemas:
            {json.dumps(available_tools_for_llm, indent=2)}
            
            Carefully consider the user's request and the capabilities of the tools.
            
            **CRITICAL INSTRUCTION FOR TOOL ARGUMENTS:**
            
            All arguments for a tool MUST be placed inside the `tool_args` dictionary within each `Step`.
            DO NOT use any other key (like 'parameters') for tool arguments.
            
            If a tool's parameter is a Pydantic model (like `RasterData` or `VectorData`), you MUST provide its fields as a DICTIONARY within `tool_args`.
            
            **Example 1: Calling a tool with a direct file path for a nested RasterData parameter:**
            ```json
            {{
                "tool_name": "reproject_raster_tool",
                "tool_args": {{
                    "input_raster": {{"path": "/data/my_image.tif", "data_type": "raster", "crs": "EPSG:4326"}},
                    "target_crs": "EPSG:3857",
                    "output_path": "/data/reprojected_image.tif",
                    "resampling_method": "bilinear"
                }},
                "output_key": "reprojected_image"
            }}
            ```
            
            **Example 2: Using a placeholder for a nested RasterData parameter where the path comes from a previous step's output:**
            ```json
            {{
                "tool_name": "reproject_raster_tool",
                "tool_args": {{
                    "input_raster": {{"path": "{{downloaded_dem.path}}", "data_type": "raster", "crs": "EPSG:4326"}},
                    "target_crs": "EPSG:3857",
                    "output_path": "reprojected_dem.tif",
                    "resampling_method": "nearest"
                }},
                "output_key": "final_reprojected_dem",
                "requires_input_from": "downloaded_dem"
            }}
            ```
            
            Always ensure 'data_type' for `RasterData` is explicitly set to 'raster' and for `VectorData` to 'vector'.
            
            Your response MUST be a single call to `create_geospatial_plan`.
            Do not include any conversational text outside of the tool call.
            """
            },
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[plan_tool_schema],
                tool_choice={"type": "function", "function": {"name": "create_geospatial_plan"}},
            )
            
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == "create_geospatial_plan":
                    plan_args = json.loads(tool_call.function.arguments)
                    plan = Plan(**plan_args)
                    logger.success("Successfully generated geospatial plan from LLM.")
                    return plan
                else:
                    raise ValueError(f"LLM called unexpected tool: {{tool_call.function.name}}")
            else:
                raise ValueError("LLM did not return a tool call for plan generation.")

        except ValidationError as e:
            logger.error(f"Pydantic validation error when creating plan: {{e.errors()}}")
            raise ValueError(f"LLM generated an invalid plan: {{e.errors()}}")
        except Exception as e:
            logger.error(f"Error generating plan with LLM: {{e}}")
            raise RuntimeError(f"Failed to generate plan: {{e}}")
