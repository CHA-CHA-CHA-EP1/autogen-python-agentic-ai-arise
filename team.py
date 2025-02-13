import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    ModelInfo,
    ModelFamily,
)
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.ui import Console


async def main():
    model_client_ollama = OpenAIChatCompletionClient(
        model="qwen2:7b",
        base_url=os.environ["OLLAMA_BASE_URL"],
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family=ModelFamily.UNKNOWN,
        ),
    )

    writer_agent = AssistantAgent(
        "writer",
        model_client=model_client_ollama,
        system_message="You're a writer, write well",
    )

    editor_agent = AssistantAgent(
        "editor",
        model_client_ollama,
        system_message="You're an editor, provide critical feedback.",
    )


asyncio.run(main())
