import json
import logging
import os
from typing import Any, Dict, List

import openai
from openai.types.chat import ChatCompletion

from geospatial_agent.domain.models import Plan
from geospatial_agent.infrastructure.config import settings

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlannerAgent:
    """
    An agent that uses an LLM to convert a user query into a structured JSON plan.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initializes the PlannerAgent.

        Args:
            model_name: The name of the OpenAI model to use.  Defaults to gpt-4o.
        """
        self.model_name = model_name
        openai.api_key = settings.openai_api_key

    def create_plan(self, query: str, tools: List[Dict[str, Any]]) -> Plan:
        """
        Converts a natural language query into a structured JSON plan using OpenAI Function Calling.

        Args:
            query: The natural language query from the user.
            tools: A list of dictionaries describing the available tools (functions).

        Returns:
            A Plan object representing the structured JSON plan.

        Raises:
            Exception: If the OpenAI API call fails.
        """
        try:
            # Call the OpenAI API with function definitions
            response: ChatCompletion = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": query}],
                functions=tools,
                function_call="auto",  # Let OpenAI decide when to use functions
            )

            message = response.choices[0].message

            # Check if the LLM decided to use a function
            if message.function_call:
                function_name = message.function_call.name
                arguments = message.function_call.arguments

                # Parse the arguments string into a JSON object
                try:
                    arguments_json = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding arguments JSON: {e}")
                    raise  # Re-raise the exception

                # Construct the plan
                plan_data = {
                    "tool": function_name,
                    "parameters": arguments_json,
                }

                plan = Plan(**plan_data)
                logger.info(f"Generated Plan: {plan}")
                return plan
            else:
                # The LLM didn't call a function.  Handle this case appropriately.
                # For now, we'll just log a warning and return None.
                logger.warning("LLM did not call a function.")
                return Plan(tool="unknown", parameters={})

        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise  # Re-raise the exception to be handled upstream