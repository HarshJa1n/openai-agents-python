from src.agents.agent import Agent
from src.agents.config_models import NewAgentConfig
from src.agents.agent_builder import build_agent_from_config, AVAILABLE_TOOLS
from src.agents.tool import Tool


def test_build_agent_with_valid_config():
    """
    Tests that an agent is correctly built with a valid NewAgentConfig.
    """
    config = NewAgentConfig(
        name="TestWeatherAgent",
        instructions="Tells weather",
        model_name="o3-mini",
        tool_names=["get_current_weather"]
    )
    agent = build_agent_from_config(config)

    assert isinstance(agent, Agent), "Agent should be an instance of Agent class."
    assert agent.name == "TestWeatherAgent", "Agent name does not match config."
    assert agent.instructions == "Tells weather", "Agent instructions do not match config."
    assert agent.model == "o3-mini", "Agent model does not match config."
    
    assert isinstance(agent.tools, list), "Agent tools should be a list."
    assert len(agent.tools) == 1, "Agent should have one tool."
    
    expected_tool = AVAILABLE_TOOLS.get("get_current_weather")
    assert agent.tools[0] == expected_tool, "The agent's tool does not match the expected tool."
    assert agent.tools[0].name == "get_current_weather", "The tool's name is incorrect."


def test_build_agent_with_non_existent_tool():
    """
    Tests agent building when config includes non-existent tool names.
    It checks that only existing tools are added and a warning is expected (though not captured here).
    """
    config = NewAgentConfig(
        name="TestMixedToolsAgent",
        instructions="Uses weather and a fake tool",
        model_name="o3-mini",
        tool_names=["get_current_weather", "non_existent_tool", "simple_calculator"]
    )
    agent = build_agent_from_config(config)

    assert isinstance(agent, Agent), "Agent should be an instance of Agent class."
    assert agent.name == "TestMixedToolsAgent"
    
    assert isinstance(agent.tools, list), "Agent tools should be a list."
    # Expecting a warning print for "non_existent_tool" from build_agent_from_config.
    # This test verifies that only the valid tools are actually added.
    assert len(agent.tools) == 2, "Agent should have only the two existing tools."
    
    expected_tool_weather = AVAILABLE_TOOLS.get("get_current_weather")
    expected_tool_calculator = AVAILABLE_TOOLS.get("simple_calculator")
    
    # Check that the correct tools are present, order might not be guaranteed
    # depending on implementation, so check for presence.
    assert expected_tool_weather in agent.tools, "get_current_weather tool missing."
    assert expected_tool_calculator in agent.tools, "simple_calculator tool missing."

    tool_names_in_agent = [tool.name for tool in agent.tools]
    assert "get_current_weather" in tool_names_in_agent
    assert "simple_calculator" in tool_names_in_agent
    assert "non_existent_tool" not in tool_names_in_agent, "Non-existent tool should not have been added."

def test_build_agent_with_no_tools():
    """
    Tests that an agent is correctly built if the config specifies no tools.
    """
    config = NewAgentConfig(
        name="TestNoToolAgent",
        instructions="This agent has no tools.",
        model_name="o3-mini",
        tool_names=[] # Empty list of tools
    )
    agent = build_agent_from_config(config)

    assert isinstance(agent, Agent)
    assert agent.name == "TestNoToolAgent"
    assert agent.instructions == "This agent has no tools."
    assert agent.model == "o3-mini"
    assert isinstance(agent.tools, list)
    assert len(agent.tools) == 0, "Agent should have no tools."

def test_build_agent_config_defaults():
    """
    Tests that an agent is correctly built using default values from NewAgentConfig
    if model_name or tool_names are not provided.
    """
    config = NewAgentConfig(
        name="TestDefaultAgent",
        instructions="This agent uses default model and tools."
        # model_name is not specified, should use default "o3-mini"
        # tool_names is not specified, should use default empty list
    )
    agent = build_agent_from_config(config)

    assert isinstance(agent, Agent)
    assert agent.name == "TestDefaultAgent"
    assert agent.instructions == "This agent uses default model and tools."
    assert agent.model == "o3-mini", "Agent model should default to o3-mini."
    assert isinstance(agent.tools, list)
    assert len(agent.tools) == 0, "Agent tools should default to an empty list."

    # Test with explicit None for tool_names (should also default to empty list due to default_factory)
    config_explicit_none_tools = NewAgentConfig(
        name="TestDefaultAgentWithNoneTools",
        instructions="Uses default model and tools, tool_names=None.",
        tool_names=None # type: ignore 
        # Pydantic v2 should handle None for optional fields with default_factory by using the factory
    )
    agent_explicit_none_tools = build_agent_from_config(config_explicit_none_tools)
    assert agent_explicit_none_tools.model == "o3-mini"
    assert len(agent_explicit_none_tools.tools) == 0

    # Test with explicit None for model_name (should use default)
    # However, model_name in NewAgentConfig has a default string, not Optional.
    # So, we rely on the default="o3-mini" in the Field definition.
    # If model_name were Optional[str] = Field(default="o3-mini"), then passing None would be different.
    # As it is, it must be a string or omitted.
    # For this test, omitting it (as done in the first part) is the way to test its default.
    # If we were to allow `model_name: Optional[str] = "o3-mini"`, Pydantic might interpret None as an explicit None value.
    # But since it's `model_name: str = "o3-mini"`, it cannot be None.
    # The default_factory for tool_names handles the None case gracefully.
    
    config_with_default_model_explicit_tools = NewAgentConfig(
        name="TestAgentDefaultModel",
        instructions="Test agent with default model and explicit tools.",
        tool_names=["simple_calculator"]
    )
    agent_with_default_model = build_agent_from_config(config_with_default_model_explicit_tools)
    assert agent_with_default_model.model == "o3-mini" # Default model
    assert len(agent_with_default_model.tools) == 1
    assert agent_with_default_model.tools[0].name == "simple_calculator"

    config_with_explicit_model_default_tools = NewAgentConfig(
        name="TestAgentDefaultTools",
        instructions="Test agent with explicit model and default tools.",
        model_name="gpt-4"
    )
    agent_with_default_tools = build_agent_from_config(config_with_explicit_model_default_tools)
    assert agent_with_default_tools.model == "gpt-4"
    assert len(agent_with_default_tools.tools) == 0 # Default empty list for tools
