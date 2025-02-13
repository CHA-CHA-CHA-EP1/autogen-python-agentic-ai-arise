import os
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    ModelInfo,
    ModelFamily,
)
from autogen_agentchat.agents import (
    AssistantAgent,
    CodeExecutorAgent,
    SocietyOfMindAgent,
)
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TokenUsageTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
import json


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
        system_message="""You are a skilled creative writer with expertise in crafting engaging and clear content. Your responsibilities include:
- Creating well-structured narratives with clear beginnings, middles, and ends
- Using vivid and precise language to convey ideas
- Maintaining consistent tone and style throughout the piece
- Adapting writing style based on the target audience and purpose
- Incorporating feedback constructively to improve your work
- Ensuring proper grammar, punctuation, and formatting
- Following standard writing conventions while maintaining creativity

""",
    )

    editor_agent = AssistantAgent(
        "editor",
        model_client_ollama,
        system_message="""You are an experienced editor with a keen eye for detail and storytelling. Your role involves:
- Providing constructive, specific feedback on both content and structure
- Identifying inconsistencies in tone, style, and narrative flow
- Suggesting improvements for clarity, conciseness, and impact
- Checking for proper grammar, punctuation, and formatting
- Ensuring content aligns with its intended purpose and audience
- Maintaining the writer's voice while enhancing the overall quality
- Offering actionable recommendations with examples when necessary
- Following standard editing conventions and style guides

Focus on being specific in your feedback, always explaining the reasoning behind your suggestions, and maintaining a balanced approach between criticism and encouragement.
Response with 'APPROVE' if the text addresses all feedback.
""",
    )

    termination = (
        TextMentionTermination(text="APPROVE")
        | MaxMessageTermination(max_messages=10)
        | TextMentionTermination(text="Approved")
    )

    team = RoundRobinGroupChat([writer_agent, editor_agent], termination)

    society_of_mind_agent = SocietyOfMindAgent(
        name="society_of_mind", team=team, model_client=model_client_ollama
    )

    translator_agent = AssistantAgent(
        name="translator",
        model_client=model_client_ollama,
        system_message="Translate the text to thai.",
    )

    final_team = RoundRobinGroupChat(
        [society_of_mind_agent, translator_agent], max_turns=2
    )

    with open("team.json", "w") as f:
        json.dump(final_team.dump_component().model_dump(), f, indent=4)

    # stream = final_team.run_stream(task="Write a short story about cat.")
    # await Console(stream)


asyncio.run(main())
