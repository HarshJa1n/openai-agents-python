from typing import List, Optional

from pydantic import BaseModel, Field


class NewAgentConfig(BaseModel):
    """Configuration for a new agent."""

    name: str = Field(..., description="Name of the agent.")
    instructions: str = Field(..., description="Instructions for the agent.")
    model_name: str = Field(
        default="o3-mini", description="Name of the model to be used."
    )
    tool_names: List[str] = Field(
        default_factory=list, description="List of tool names to be used."
    )
