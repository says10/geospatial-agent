from typing import Any, Dict

from pydantic import BaseModel


class Plan(BaseModel):
    """
    Represents a structured execution plan.
    """

    tool: str
    parameters: Dict[str, Any]