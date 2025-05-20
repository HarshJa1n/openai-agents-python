import asyncio

from src.agents.runner import Runner
from src.agents.meta_agent import MetaAgent
from src.agents.config_models import NewAgentConfig
from src.agents.agent_builder import AVAILABLE_TOOLS


async def test_meta_agent_produces_valid_config():
    """
    Tests that the MetaAgent can process a user request and produce a valid NewAgentConfig.
    Note: This test interacts with an LLM and may require the OPENAI_API_KEY environment
    variable to be set if MetaAgent uses an OpenAI model. It also assumes the LLM
    can follow the instructions to populate the NewAgentConfig correctly.
    """
    user_request = (
        "Create an agent named 'CalculatorPro' that can calculate things using the "
        "'simple_calculator' tool and uses the 'gpt-4' model. "
        "It should be very good at math and assist users with their calculations."
    )
    
    prompt_to_meta_agent = (
        f"Available tools: {list(AVAILABLE_TOOLS.keys())}\n\n"
        f"User request: {user_request}"
    )

    # Ensure OPENAI_API_KEY is set in environment for this test to run if using OpenAI models.
    run_result = await Runner.run(
        agent=MetaAgent,
        input_data=prompt_to_meta_agent,
        # verbose=True # Uncomment for detailed logging during debugging
    )

    assert run_result is not None, "Runner.run did not return a result."
    assert run_result.error is None, f"MetaAgent run failed with error: {run_result.error}"
    
    config_output = run_result.final_output_as(NewAgentConfig)

    assert config_output is not None, \
        f"MetaAgent did not produce a parsable NewAgentConfig. Raw output: {run_result.raw_output}"
    assert isinstance(config_output, NewAgentConfig), \
        "The output from MetaAgent is not an instance of NewAgentConfig."

    # Name: Expecting something like "CalculatorPro" but LLM might vary it.
    # Check for non-empty and containing the core name.
    assert isinstance(config_output.name, str), "Config name should be a string."
    assert len(config_output.name) > 0, "Config name should not be empty."
    assert "CalculatorPro" in config_output.name, \
        f"Expected 'CalculatorPro' in agent name, got: {config_output.name}"

    # Instructions: Should be a non-empty string.
    assert isinstance(config_output.instructions, str), "Config instructions should be a string."
    assert len(config_output.instructions) > 0, "Config instructions should not be empty."
    assert "math" in config_output.instructions.lower() or \
           "calculate" in config_output.instructions.lower(), \
           f"Expected instructions to be related to math/calculations, got: {config_output.instructions}"


    # Model Name: Expecting "gpt-4" as requested.
    assert isinstance(config_output.model_name, str), "Config model_name should be a string."
    assert config_output.model_name == "gpt-4", \
        f"Expected model_name 'gpt-4', got: {config_output.model_name}"

    # Tool Names: Expecting ["simple_calculator"].
    assert isinstance(config_output.tool_names, list), "Config tool_names should be a list."
    assert "simple_calculator" in config_output.tool_names, \
        f"Expected 'simple_calculator' in tool_names, got: {config_output.tool_names}"
    
    # Ensure only valid tools are listed if LLM tries to add more.
    for tool_name in config_output.tool_names:
        assert tool_name in AVAILABLE_TOOLS, f"Unknown tool '{tool_name}' listed by MetaAgent."


async def test_meta_agent_handles_no_specific_tools_request():
    """
    Tests MetaAgent's behavior when the user request doesn't specify any tools.
    It should produce a config with an empty tool_names list.
    """
    user_request = (
        "I need a friendly assistant called 'ChattyHelper' that can talk about general topics. "
        "It should use the o3-mini model."
    )
    prompt_to_meta_agent = (
        f"Available tools: {list(AVAILABLE_TOOLS.keys())}\n\n"
        f"User request: {user_request}"
    )

    run_result = await Runner.run(MetaAgent, prompt_to_meta_agent)
    assert run_result.error is None, f"MetaAgent run failed: {run_result.error}"

    config_output = run_result.final_output_as(NewAgentConfig)
    assert config_output is not None, f"MetaAgent did not produce NewAgentConfig. Raw: {run_result.raw_output}"
    
    assert "ChattyHelper" in config_output.name
    assert config_output.model_name == "o3-mini" # or MetaAgent's default if not overridden by LLM
    assert isinstance(config_output.tool_names, list)
    assert len(config_output.tool_names) == 0, \
        f"Expected no tools for ChattyHelper, but got: {config_output.tool_names}"

async def test_meta_agent_uses_default_model_if_not_specified():
    """
    Tests that MetaAgent uses its default model (or a reasonable default like 'o3-mini'
    as specified in NewAgentConfig) if the user doesn't specify one.
    """
    user_request = (
        "Create an agent named 'QuickNoteTaker'. It should just take notes. "
        "It needs the 'simple_calculator' tool, surprisingly for a note taker!" # test tool selection
    )
    prompt_to_meta_agent = (
        f"Available tools: {list(AVAILABLE_TOOLS.keys())}\n\n"
        f"User request: {user_request}"
    )
    run_result = await Runner.run(MetaAgent, prompt_to_meta_agent)
    assert run_result.error is None, f"MetaAgent run failed: {run_result.error}"

    config_output = run_result.final_output_as(NewAgentConfig)
    assert config_output is not None, f"MetaAgent did not produce NewAgentConfig. Raw: {run_result.raw_output}"

    assert "QuickNoteTaker" in config_output.name
    # MetaAgent is configured with o3-mini, NewAgentConfig also defaults to o3-mini.
    # The LLM should pick this up from its instructions or the NewAgentConfig structure.
    assert config_output.model_name == "o3-mini", \
        f"Expected default model 'o3-mini', got: {config_output.model_name}"
    assert "simple_calculator" in config_output.tool_names
    assert len(config_output.tool_names) == 1
