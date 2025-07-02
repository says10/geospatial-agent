import inspect
import json
from typing import get_type_hints, Union, Any, List
from docstring_parser import parse

# --- Impor tool functions ---
from .buffer import buffer_layer
from .intersect import intersect_layers
from .clip import clip_layer


def map_type_to_json_schema(py_type: Any) -> str:
    """Maps Python type hints to JSON Schema types."""
    # This handles Union types, e.g., Union[str, Path] -> str
    if hasattr(py_type, '__origin__') and py_type.__origin__ is Union:
        # For simplicity, we'll take the first non-None type in the Union
        py_type = next(t for t in py_type.__args__ if t is not type(None))

    if py_type is str or py_type.__name__ == 'Path':
        return "string"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    return "string"  # Default to string for unknown types

def generate_tool_definitions(tool_functions: List[callable]) -> str:
    """
    Generates JSON definitions for a list of tool functions.

    Args:
        tool_functions: A list of function objects to be documented.

    Returns:
        A JSON formatted string containing the definitions of all tools.
    """
    tool_definitions = []
    for func in tool_functions:
        # Parse the docstring to get descriptions
        docstring = parse(inspect.getdoc(func))
        
        # Get function signature for parameters and type hints
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param in signature.parameters.values():
            if param.name == 'self':  # Skip self in class methods
                continue
            
            param_type = type_hints.get(param.name, str)
            
            # Find the parameter's description in the parsed docstring
            param_doc = next((d for d in docstring.params if d.arg_name == param.name), None)
            param_description = param_doc.description if param_doc else ""

            properties[param.name] = {
                "type": map_type_to_json_schema(param_type),
                "description": param_description
            }

            # If the parameter has no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(param.name)

        tool_def = {
            "name": func.__name__,
            "description": docstring.short_description + (" " + docstring.long_description if docstring.long_description else ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
        tool_definitions.append(tool_def)

    return json.dumps(tool_definitions, indent=2)

if __name__ == '__main__':
    # --- Define the list of all your agent's tools ---
    all_tools = [
        buffer_layer,
        intersect_layers,
        clip_layer
        # Add other imported tools here
    ]

    # Generate the JSON string
    tools_json = generate_tool_definitions(all_tools)

    # Print it or save it to a file
    print(tools_json)

    # To save to a file:
    # with open('prompt_tools.json', 'w') as f:
    #     f.write(tools_json)