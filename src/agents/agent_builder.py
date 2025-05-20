from typing import Optional, List, Dict

from src.agents.agent import Agent
from src.agents.config_models import NewAgentConfig
from src.agents.tool import Tool, function_tool


@function_tool(description_override="Gets the current weather for a specified location.")
def get_current_weather(location: str, unit: Optional[str] = "celsius") -> str:
    """
    Simulates fetching the current weather for a given location.
    """
    return f"The weather in {location} is sunny and 25 {unit}."


@function_tool(description_override="A simple calculator that can evaluate basic arithmetic expressions (+, -, *, /).")
def simple_calculator(expression: str) -> str:
    """
    Evaluates a simple arithmetic expression.
    Important Security Note: eval() can be dangerous with untrusted input.
    In a real-world scenario, a proper expression parser should be used.
    This example only allows basic arithmetic operations for safety.
    """
    allowed_chars = "0123456789+-*/(). "
    if not all(char in allowed_chars for char in expression):
        return "Error: Expression contains invalid characters."
    try:
        # Security Note: eval() is used here for simplicity but is dangerous with untrusted input.
        # A production system should use a safer method to parse and evaluate expressions.
        result = eval(expression)
        return f"The result of '{expression}' is {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


AVAILABLE_TOOLS: Dict[str, Tool] = {
    "get_current_weather": get_current_weather,
    "simple_calculator": simple_calculator,
}


def build_agent_from_config(config: NewAgentConfig) -> Agent:
    """
    Builds an agent from a NewAgentConfig instance.

    Args:
        config: The configuration for the new agent.

    Returns:
        An Agent instance.
    """
    resolved_tools: List[Tool] = []
    for tool_name in config.tool_names:
        tool = AVAILABLE_TOOLS.get(tool_name)
        if tool:
            resolved_tools.append(tool)
        else:
            print(f"Warning: Tool '{tool_name}' not found in AVAILABLE_TOOLS and will be skipped.")

    agent = Agent(
        name=config.name,
        instructions=config.instructions,
        model=config.model_name,  # Agent's 'model' parameter takes the model name string
        tools=resolved_tools,
        # Other parameters like output_type, handoffs can be defaults or configured if needed
    )
    return agent
