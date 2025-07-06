# test_agent.py

import os
from pathlib import Path
import asyncio
from loguru import logger
import json
import sys # Import sys to inspect sys.path

# --- DEBUGGING PRINTS ---
print(f"Current working directory: {os.getcwd()}")
project_root_calculated = Path(__file__).resolve().parent
print(f"Calculated project root: {project_root_calculated}")
src_path_to_add = str(project_root_calculated / "src")
print(f"Path to add to sys.path: {src_path_to_add}")

if src_path_to_add not in sys.path:
    sys.path.insert(0, src_path_to_add)
    print(f"Added '{src_path_to_add}' to sys.path.")
else:
    print(f"'{src_path_to_add}' already in sys.path.")

print("\n--- Current sys.path ---")
for p in sys.path:
    print(p)
print("------------------------\n")
# --- END DEBUGGING PRINTS ---


# Ensure the project root is in the Python path for imports
# This assumes test_agent.py is in the root of the geospatial-agent directory
# project_root = Path(__file__).resolve().parent # Already calculated above for debugging
# import sys # Already imported above for debugging
# if str(project_root / "src") not in sys.path:
#     sys.path.insert(0, str(project_root / "src"))


from geospatial_agent.application.agents.planner_agent import PlannerAgent
from geospatial_agent.tool_manager.tool_manager import ToolManager

logger.add("file_log.log", rotation="500 MB") # Log to a file
logger.info("Starting geospatial agent test script.")

async def main():
    """
    Main function to test the PlannerAgent and ToolManager.
    """
    # Ensure data directories exist for outputs
    data_dir = project_root_calculated / "data" # Use the calculated project_root
    downloads_dir = data_dir / "downloads"
    tool_outputs_dir = data_dir / "tool_outputs"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    tool_outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured data directories exist: {downloads_dir}, {tool_outputs_dir}")

    tool_manager = ToolManager()
    planner_agent = PlannerAgent(tool_manager=tool_manager)

    test_queries = [
        "I need to download a DEM for the region bounded by longitude 72 to 75 and latitude 12 to 15, and save it as 'my_dem.tif'.",
        "Convert the vector file 'data/downloads/dummy_aoi.geojson' to a Shapefile named 'my_aoi.shp'.",
        "Clip the raster 'data/downloads/dummy_dem.tif' using the vector 'data/downloads/dummy_aoi.geojson' and save the output as 'clipped_dem.tif'.",
        "Calculate pixel statistics for the first band of 'data/downloads/dummy_dem.tif'.",
        "What is the CRS and dimensions of the raster file 'data/downloads/dummy_dem.tif'?"
    ]

    for i, query in enumerate(test_queries):
        logger.info(f"\n--- Test Case {i+1}: {query} ---")
        try:
            # Step 1: Generate Plan
            logger.info("Attempting to generate plan...")
            plan = planner_agent.generate_plan(query)
            logger.info(f"Generated Plan (Goal: {plan.goal}, Final Output Key: {plan.final_output_key}):\n{json.dumps(plan.model_dump(), indent=2)}")

            # Step 2: Execute Plan
            logger.info("Attempting to execute plan steps...")
            execution_results = {}
            for step in plan.steps:
                tool_name = step.tool_name
                tool_args = step.tool_args.copy() # Make a copy to avoid modifying original plan
                output_key = step.output_key
                requires_input_from = step.requires_input_from

                # Resolve dynamic inputs from previous step results
                if requires_input_from and requires_input_from in execution_results:
                    prev_result = execution_results[requires_input_from]
                    # Assuming the previous result has a 'path' attribute or 'path' key
                    resolved_path = getattr(prev_result, 'path', prev_result.get('path'))

                    # Iterate through tool_args to replace placeholders
                    for arg_key, arg_value in tool_args.items():
                        if isinstance(arg_value, dict) and "path" in arg_value and isinstance(arg_value["path"], str):
                            # Replace {output_key.path} with actual path
                            if f"{{{requires_input_from}.path}}" in arg_value["path"]:
                                tool_args[arg_key]["path"] = arg_value["path"].replace(f"{{{requires_input_from}.path}}", resolved_path)
                                logger.debug(f"Resolved dynamic input for {arg_key}: {tool_args[arg_key]['path']}")
                        elif isinstance(arg_value, str) and f"{{{requires_input_from}.path}}" in arg_value:
                            tool_args[arg_key] = arg_value.replace(f"{{{requires_input_from}.path}}", resolved_path)
                            logger.debug(f"Resolved dynamic input for {arg_key}: {tool_args[arg_key]}")

                logger.info(f"Executing tool '{tool_name}' for step '{output_key}' with args: {tool_args}")
                step_result = tool_manager.execute_tool(tool_name, **tool_args)
                execution_results[output_key] = step_result
                logger.info(f"Step '{output_key}' result: {step_result}")

            final_result = execution_results.get(plan.final_output_key)
            logger.success(f"Plan execution complete. Final result for '{plan.final_output_key}': {final_result}")

        except Exception as e:
            logger.error(f"Error during test case {i+1}: {e}")
            logger.exception("Full traceback for the error:") # Log full traceback

if __name__ == "__main__":
    asyncio.run(main())