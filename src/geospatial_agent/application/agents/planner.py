import json
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.runnables import chain
from pydantic import BaseModel, Field

from geospatial_agent.infrastructure.rag.ontology import load_ontology
from geospatial_agent.infrastructure.config import Config

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define Pydantic model for the plan
class Plan(BaseModel):
    """
    A plan is a series of steps to be executed.
    """
    steps: list[Dict[str, Any]] = Field(
        ...,
        description="A list of steps to be executed in order. Each step should include function_name and parameters.",
    )

# Load ontology
ontology = load_ontology(Config.ONTOLOGY_FILE_PATH)

# Build tool descriptions for prompt
tool_descriptions = "\n".join(
    [
        f"- {tool['name']}: {tool['description']}. Parameters: {tool['parameters']}"
        for tool in ontology["tools"]
    ]
)

# Define the prompt template
SYSTEM_TEMPLATE = f"""
You are a helpful AI agent that generates execution plans for geospatial tasks.
You have access to the following tools:
{tool_descriptions}
Given a user query, create a JSON plan consisting of steps to accomplish the task.
The plan should be a valid JSON object that is a list of dictionaries.
Given a user query, create a JSON plan consisting of steps to accomplish the task.
The plan MUST be a valid JSON object that conforms to the following schema:
```json
{json.dumps(Plan.model_json_schema())}
Each dictionary should have the following keys: "function_name" and "parameters".
The "function_name" key should be one of the available tools.
The "parameters" key should be a dictionary of parameters required for the tool.
The plan should be as concise as possible.
Do not add any explanations or other text.
"""

class PlannerAgent:
    """
    An agent that uses an LLM to generate execution plans for geospatial tasks.
    """

    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initializes the PlannerAgent.
        Args:
            model_name: The name of the OpenAI model to use. Defaults to "gpt-4o".
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=Config.OPENAI_API_KEY)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_TEMPLATE),
                HumanMessagePromptTemplate.from_template("User query: {query}"),
            ]
        )
        self.chain = self.prompt | self.llm

    def create_plan(self, query: str) -> Plan:
        """
        Creates an execution plan for the given query.
        Args:
            query: The user query.
        Returns:
            A Plan object representing the execution plan.
        """
        try:
            parser = PydanticOutputParser(pydantic_object=Plan)
            prompt = self.prompt.format_messages(query=query)
            response = self.llm(prompt)
            logger.info(f"Raw LLM output: {response.content}") 
            plan = parser.parse(response.content)
            logger.info(f"Generated plan: {plan}")
            return plan
        except Exception as e:
            logger.exception("Error creating plan:")
            raise

if __name__ == '__main__':
    planner = PlannerAgent()
    query = "Calculate the area of a plot of land, correcting for terrain slope."  # Modified Query
    plan = planner.create_plan(query)
    print(plan.model_dump_json(indent=2))