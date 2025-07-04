import pytest
from geospatial_agent.application.agents.planner import PlannerAgent

def test_planner_create_plan():
    planner = PlannerAgent()
    query = "Calculate the area of a plot of land, correcting for terrain slope."
    plan = planner.create_plan(query)

    assert plan is not None
    assert isinstance(plan.steps, list)
    assert len(plan.steps) > 0  # Expect at least one step in the plan

    # Add more specific assertions based on expected plan structure.  For example:
    function_names = [step["function_name"] for step in plan.steps]
    assert "calculate_area" in function_names
    assert "read_shapefile" in function_names

    # you'll need to read the shapefile before calculating area
    calculate_area_step = next((step for step in plan.steps if step["function_name"] == "calculate_area"), None)
    assert calculate_area_step is not None
    assert "slope_corrected" in calculate_area_step["parameters"]