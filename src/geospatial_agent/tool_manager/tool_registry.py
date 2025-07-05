# src/geospatial_agent/tool_manager/tool_registry.py

from typing import Dict, Callable, Any, Optional, List
from loguru import logger

# Import the ToolDefinition ontology model
from geospatial_agent.ontology.ontology import ToolDefinition

class ToolRegistry:
    """
    Manages a registry of available tools (functions) and their corresponding
    ToolDefinition metadata.

    This class serves as a central catalog for the geospatial agent, allowing it
    to discover and access tools based on their defined capabilities.
    """

    def __init__(self):
        """
        Initializes an empty ToolRegistry.
        """
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(self, tool_definition: ToolDefinition, tool_function: Callable[..., Any]):
        """
        Registers a tool function along with its metadata (ToolDefinition).

        Args:
            tool_definition (ToolDefinition): A Pydantic model defining the tool's capabilities.
            tool_function (Callable[..., Any]): The actual Python function implementing the tool.
        """
        tool_name = tool_definition.name
        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' is already registered. Overwriting existing registration.")

        self._tools[tool_name] = tool_function
        self._tool_definitions[tool_name] = tool_definition
        logger.info(f"Tool '{tool_name}' registered successfully.")

    def get_tool(self, tool_name: str) -> Optional[Callable[..., Any]]:
        """
        Retrieves a registered tool function by its name.

        Args:
            tool_name (str): The name of the tool to retrieve.

        Returns:
            Optional[Callable[..., Any]]: The tool function if found, otherwise None.
        """
        tool_function = self._tools.get(tool_name)
        if tool_function is None:
            logger.warning(f"Attempted to retrieve unregistered tool: '{tool_name}'.")
        return tool_function

    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Retrieves a registered tool's definition by its name.

        Args:
            tool_name (str): The name of the tool's definition to retrieve.

        Returns:
            Optional[ToolDefinition]: The tool definition if found, otherwise None.
        """
        tool_definition = self._tool_definitions.get(tool_name)
        if tool_definition is None:
            logger.warning(f"Attempted to retrieve definition for unregistered tool: '{tool_name}'.")
        return tool_definition

    def get_all_tool_definitions(self) -> List[ToolDefinition]:
        """
        Retrieves a list of all registered tool definitions.

        Returns:
            List[ToolDefinition]: A list of all ToolDefinition objects.
        """
        return list(self._tool_definitions.values())

    def clear_registry(self):
        """
        Clears all registered tools and their definitions from the registry.
        Primarily for testing or re-initialization.
        """
        self._tools.clear()
        self._tool_definitions.clear()
        logger.info("ToolRegistry cleared.")
