# To run this example from the project root, you might need to set your PYTHONPATH:
# export PYTHONPATH=.
# Then run:
# python examples/meta_agent_builder/run_builder.py
#
# Alternatively, if the 'openai-agents' package is installed (e.g., `pip install -e .`),
# direct imports might work depending on your environment setup.

import asyncio

from src.agents.runner import Runner
from src.agents.meta_agent import MetaAgent
from src.agents.agent_builder import build_agent_from_config, AVAILABLE_TOOLS
from src.agents.config_models import NewAgentConfig


async def main():
    """
    Demonstrates using MetaAgent to create a new agent based on a user request,
    and then running that newly built agent.
    """
    print(f"Available tools for MetaAgent to choose from: {list(AVAILABLE_TOOLS.keys())}")

    user_request_for_new_agent = (
        "I need an agent that can tell me the weather and calculate simple math. "
        "Call it 'WeatherMathHelper'. It should be very friendly and use the 'get_current_weather' and 'simple_calculator' tools."
    )
    print(f"\nUser request for new agent: \"{user_request_for_new_agent}\"")

    print(f"\nUsing MetaAgent: {MetaAgent.name} (Model: {MetaAgent.model})")
    print("MetaAgent Instructions:\n---")
    print(MetaAgent.instructions)
    print("---\n")

    # Ensure OPENAI_API_KEY is set in your environment variables if MetaAgent uses an OpenAI model.
    print("Running MetaAgent to generate agent configuration...")
    meta_agent_run_result = await Runner.run(
        agent=MetaAgent,
        input_data=user_request_for_new_agent,
        # verbose=True # Uncomment for more detailed logging from the runner
    )

    if meta_agent_run_result.error:
        print(f"MetaAgent run failed: {meta_agent_run_result.error}")
        return

    # The MetaAgent's output_type is NewAgentConfig, so we can directly get it.
    generated_config = meta_agent_run_result.final_output_as(NewAgentConfig)

    if not generated_config:
        print("MetaAgent did not produce a valid NewAgentConfig.")
        if meta_agent_run_result.raw_output:
            print(f"Raw output from MetaAgent: {meta_agent_run_result.raw_output}")
        return

    print(f"\nMetaAgent generated config:\n{generated_config.model_dump_json(indent=2)}")

    # 2. Build the new agent using the generated configuration
    print("\nBuilding the new agent from the generated config...")
    newly_built_agent = build_agent_from_config(generated_config)
    print(f"Successfully built agent: {newly_built_agent.name} (Model: {newly_built_agent.model})")
    print(f"Instructions for {newly_built_agent.name}:\n---")
    print(newly_built_agent.instructions)
    print("---")
    print(f"Tools for {newly_built_agent.name}: {[tool.name for tool in newly_built_agent.tools]}")


    # 3. Run the newly built agent
    input_for_new_agent = "What's the weather in London and what is 5 + 7 multiplied by 2?"
    print(f"\nRunning the newly built agent '{newly_built_agent.name}' with input: \"{input_for_new_agent}\"")

    # Ensure OPENAI_API_KEY is set if the newly_built_agent uses an OpenAI model.
    new_agent_run_result = await Runner.run(
        agent=newly_built_agent,
        input_data=input_for_new_agent,
        # verbose=True # Uncomment for more detailed logging
    )

    if new_agent_run_result.error:
        print(f"Newly built agent run failed: {new_agent_run_result.error}")
        return

    print(f"\nFinal output from '{newly_built_agent.name}':")
    print(new_agent_run_result.final_output)


if __name__ == "__main__":
    # Note: OPENAI_API_KEY environment variable must be set for this script
    # if the agents are using OpenAI models.
    # You can set it by running `export OPENAI_API_KEY='your_key_here'` in your terminal
    # or by using a .env file with a library like python-dotenv.
    # For example, you could add:
    # import os
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Warning: OPENAI_API_KEY is not set. Agent runs requiring OpenAI models may fail.")
    #     print("Please set it in your environment: export OPENAI_API_KEY='your_key_here'")
    #     # Potentially exit or use a mock API key if appropriate for the example.
    asyncio.run(main())
