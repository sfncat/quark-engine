# -*- coding: utf-8 -*-
# This file is part of Quark-Engine - https://github.com/quark-engine/quark-engine
# See the file 'LICENSE' for copying permission.

import os
import click
from quark.utils.colors import green, cyan


def __printDependencyMissingMessage() -> None:
    print("Quark Agent requires langchain and its API integrations to work.")
    print(
        (
            "Please use the command 'python3 -m pip install"
            " langchain langchain-core langchain-openai langchain-deepseek --upgrade'"
            " to install the packages."
        )
    )


def __setOrAskAPIKey(apiKey: str, provider: str) -> bool:
    env_key = f"{provider.upper()}_API_KEY"
    if apiKey:
        os.environ[env_key] = apiKey
    elif env_key not in os.environ:
        try:
            os.environ[env_key] = click.prompt(
                f"Please provide the access key of {provider} API"
            )
        except click.Abort:
            return False

    return True


@click.command()
@click.option(
    "--api-key",
    help="Access key of API provider",
    type=str,
    show_default=False,
    default=None,
)
@click.option(
    "--provider",
    help="API provider (openai or deepseek)",
    type=click.Choice(["openai", "deepseek"]),
    show_default=True,
    default="openai",
)
def entryPoint(api_key: str, provider: str) -> None:

    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor
        from langchain_core.prompts import (
            ChatPromptTemplate,
            MessagesPlaceholder,
        )
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain.agents.output_parsers.openai_tools import (
            OpenAIToolsAgentOutputParser,
        )
        from langchain.agents.format_scratchpad.openai_tools import (
            format_to_openai_tool_messages,
        )
    except ModuleNotFoundError:
        __printDependencyMissingMessage()
        # langchain is not installed.
        return

    from quark.agent.agentTools import agentTools
    from quark.agent.prompts import SUMMARY_REPORT_FORMAT

    if not __setOrAskAPIKey(api_key, provider):
        # API Key is not provided.
        return

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    else:
        from langchain_deepseek import ChatDeepseek
        llm = ChatDeepseek(model="deepseek-chat", temperature=0.8)

    llmWithTools = llm.bind_tools(agentTools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are very powerful assistant, "
                    + "but don't know current events"
                )
                + SUMMARY_REPORT_FORMAT,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llmWithTools
        | OpenAIToolsAgentOutputParser()
    )

    agentExecutor = AgentExecutor(agent=agent, tools=agentTools, verbose=False)

    conversationHistory = []

    try:
        inputText = input(green("User Input: "))
        while inputText.lower() != "bye":
            if inputText:
                response = agentExecutor.invoke(
                    {"input": inputText, "chat_history": conversationHistory}
                )

                conversationHistory.extend(
                    [
                        HumanMessage(content=inputText),
                        AIMessage(content=response["output"]),
                    ]
                )

                print()
                print(cyan("Agent: "), response["output"])
                print()

            inputText = input(green("User Input: "))
    except click.Abort:
        return


if __name__ == "__main__":
    entryPoint()  # pylint: disable=E1120
