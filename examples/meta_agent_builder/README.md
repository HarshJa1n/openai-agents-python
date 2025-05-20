# Meta Agent Builder Example

This example demonstrates the capabilities of the "Meta Agent Builder" system.

## `run_builder.py`

The `run_builder.py` script showcases the end-to-end process of:

1.  **Defining a User Request**: A natural language prompt is created to describe a new desired agent (e.g., its name, purpose, required tools).
2.  **Using `MetaAgent`**: The `MetaAgent` (from `src.agents.meta_agent`) processes this user request.
3.  **Generating Agent Configuration**: The `MetaAgent` outputs a `NewAgentConfig` object (defined in `src.agents.config_models`). This configuration specifies the name, instructions, AI model, and tools for the new agent.
4.  **Building the New Agent**: The `build_agent_from_config` function (from `src.agents.agent_builder`) takes the `NewAgentConfig` and constructs an actual `Agent` instance.
5.  **Running the New Agent**: The newly built agent is then run with a sample input to demonstrate its functionality.

**To run this example:**

Ensure you are in the project root directory. You may need to set your `PYTHONPATH`:

```bash
export PYTHONPATH=.
```

Then execute the script:

```bash
python examples/meta_agent_builder/run_builder.py
```

**Important**: This example may use OpenAI models via the `MetaAgent` or the agents it creates. If so, you must have the `OPENAI_API_KEY` environment variable set to your OpenAI API key.

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Future Enhancements

The Meta Agent Builder concept has significant potential for expansion. Some future enhancements could include:

*   **Dynamic Tool Creation**: Allowing the `MetaAgent` to not only select existing tools but also define or generate code for new tools based on user descriptions.
*   **Advanced Agent Configuration**: Extending `NewAgentConfig` and the `MetaAgent`'s capabilities to configure more complex agent features, such as:
    *   Specific output types for the new agent (e.g., Pydantic models).
    *   Inter-agent handoff configurations.
    *   Detailed model settings (e.g., temperature, max tokens).
*   **Configuration Persistence**: Implementing a way to save a generated `NewAgentConfig` to a file (e.g., JSON/YAML) and load it later to recreate an agent, facilitating reusability and sharing of agent designs.
*   **Interactive Building Flow**: Developing a more conversational interface for agent creation, where users can iteratively refine the agent's configuration through a dialogue with the `MetaAgent`.
*   **Expanded Tool Registry**: Introducing mechanisms for users to register their own custom tools, making them available for the `MetaAgent` to include in new agent configurations.
