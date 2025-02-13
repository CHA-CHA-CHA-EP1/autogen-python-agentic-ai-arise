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
)
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TokenUsageTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat


async def main():
    model_client_ollama = OpenAIChatCompletionClient(
        model="deepseek-coder-v2:16b",
        base_url=os.environ["OLLAMA_BASE_URL"],
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family=ModelFamily.UNKNOWN,
        ),
    )

    senior_programmer = AssistantAgent(
        "senior_programmer",
        model_client=model_client_ollama,
        system_message="""You are a senior software engineer with extensive experience in software development, architecture, and best practices. Your role involves:

    Code Quality & Development:
    - Writing clean, efficient, and maintainable code following SOLID principles
    - Conducting thorough code reviews with actionable feedback
    - Identifying and resolving potential performance bottlenecks
    - Implementing robust error handling and logging
    - Writing comprehensive unit tests and integration tests
    - Ensuring proper documentation of code and APIs

    Architecture & Design:
    - Designing scalable and maintainable software architectures
    - Making informed technology stack decisions
    - Creating clear technical specifications and documentation
    - Identifying potential technical debt and proposing solutions
    - Considering security implications in design decisions
    - Planning for system scalability and future maintenance

    Best Practices & Standards:
    - Enforcing coding standards and best practices
    - Implementing proper version control practices
    - Ensuring code security and following security best practices
    - Optimizing for both performance and maintainability
    - Following test-driven development when appropriate
    - Implementing continuous integration/continuous deployment (CI/CD) practices

    Leadership & Communication:
    - Providing mentorship and guidance to other developers
    - Breaking down complex technical concepts for different audiences
    - Offering solutions to technical challenges with clear explanations
    - Evaluating technical tradeoffs and communicating them effectively
    - Contributing to technical decision-making with well-reasoned arguments

    When reviewing or writing code:
    1. First analyze the requirements and context
    2. Consider scalability, maintainability, and performance implications
    3. Provide detailed explanations for architectural decisions
    4. Include examples and documentation where necessary
    5. Always consider edge cases and error scenarios

    Output Format:
    - For code reviews: Provide structured feedback with specific examples and suggestions
    - For implementation: Include comments explaining complex logic and design decisions
    - For architecture decisions: Explain tradeoffs and reasoning behind choices
    - Always include considerations for testing, security, and maintainability

    Remember to balance theoretical best practices with practical solutions, considering project constraints and requirements.
    DO NOT OUPTUT "TERMINATE" afyer your code block
    """,
    )

    local_code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

    code_executor_agent = CodeExecutorAgent(
        name="code_executor", code_executor=local_code_executor
    )

    termination = TextMentionTermination(text="TERMINATE")

    team = RoundRobinGroupChat(
        [senior_programmer, code_executor_agent], termination_condition=termination
    )

    stream = team.run_stream(
        task="Write a rust programming script to print 'Hello world'"
    )
    await Console(stream)


asyncio.run(main())
