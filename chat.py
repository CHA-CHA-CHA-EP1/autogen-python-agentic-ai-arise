import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    ModelInfo,
    ModelFamily,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)


async def main():
    model_client_ollama = OpenAIChatCompletionClient(
        model="qwen2:7b",
        base_url=os.environ["OLLAMA_BASE_URL"],
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family=ModelFamily.UNKNOWN,
        ),
    )

    messages = []
    messages.append(SystemMessage(content="You are a helpful personal assistant"))
    while True:
        user_message = input("User: ")
        if user_message == "exit":
            break

        messages.append(UserMessage(content=user_message, source="user"))
        response = await model_client_ollama.create(messages=messages)
        print(f"{response.content}\n{response.usage}")
        messages.append(AssistantMessage(content=response.content, source="assistant"))


asyncio.run(main())
