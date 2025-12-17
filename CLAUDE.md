you're in a uv environment, so make sure you run python commands etc with "uv run ..."
agents work like this now with v1+ langchain:
```
from langchain.agents import create_agent

simple_agent = create_agent(
    llm,
    tools=simple_tools,
    system_prompt=AGENT_SYSTEM_PROMPT,
)```
