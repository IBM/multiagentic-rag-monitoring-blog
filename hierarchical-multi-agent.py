import os
from dotenv import load_dotenv
from typing import Annotated, List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict
from typing import List, Optional, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from milvus import run_search
from traceloop.sdk.decorators import workflow

#========================== Load the environment variables =====================
load_dotenv()
# Load environment variables
apikey = os.getenv("WATSONX_API_KEY")
url = os.getenv("WATSONX_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
tavily_api_key = os.getenv("TAVILY_API_KEY")


#========================== Observability with Instana =====================

from traceloop.sdk import Traceloop
Traceloop.init(app_name="AgenticRAG", api_endpoint="localhost:4317")


#========================== Set working directory to save files =====================
WORKING_DIRECTORY = Path("files")

# #========================== Define LLM =====================
rits_api_key = os.getenv("RITS_API_KEY")

llm = ChatOpenAI(
    model= "ibm-granite/granite-3.1-8b-instruct", #"meta-llama/llama-3-3-70b-instruct", #"ibm-granite/granite-3.1-2b-instruct", #,
    temperature=0,
    max_retries=2,
    api_key='/',
    base_url= 'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1', #'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct', #'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-2b-instruct', #'https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1',
    default_headers={'RITS_API_KEY': str(rits_api_key)},
)

# #========================== Tools =====================

tavily_tool = TavilySearchResults(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

@tool
def retrieve_recent_events_rag(query: str) -> str:
    """Use this for events related to top best AI agents"""
    results = run_search(query)
    return results


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY/file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

# ========================== Supervisor node ===================================

class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members

    system_prompt = (
    "You are a supervisor managing a structured workflow involving {members}. "
    "Your role is to efficiently delegate tasks and coordinate actions based on the user's request."
    "\n\n"
    "**Task Progression Rules:**\n"
    "1. If the query asks to research, go to the `research_team`."
    "2. Once the `research_team` has `FINISH`, and the query has write a `document`, `blog`, `note`, or `generate a chart`, go to the `writing_team`."
    "3. Only respond with 'FINISH' when ALL tasks are complete and there are no further actions to be taken."
    "\n"
    "- DO NOT enter an infinite loop of research or writing. Each task should transition smoothly into the next phase (e.g., from research to writing or finalizing)."
    "- Always prioritize progress toward completing the final output (document, blog, etc.). Only conclude when no more tasks remain."
    "\n"
    "What should be the next action?"
)


    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options] # * unpacks list options

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # Check if the task has been performed or not to avoid infinite loops
        if state.get("task_done", False):  # If task is done, stop
            return Command(goto=END, update={"next": "FINISH"})

        # Output should be in the structure dictated by the Router class
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]

        # Return the appropriate next node
        return Command(goto=goto, update={"next": goto})

    return supervisor_node

# ========================== Research Team ===================================

# Research Supervisor Node

def make_research_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
    "You are a research supervisor managing a structured workflow involving {members}. "
    "Your role is to assign tasks efficiently and progress through a series of steps based on the user's request."
    "\n\n"
    "**Task Progression Rules:**\n"
    "1. If the query has `top AI agents` in it, call `rag_agent` first and limit your response to at most one or two calls to `rag_agent`.\n"
    "2. If the query does not have `top AI agents` in it, limit calls to `search` or to `web_scraper` to at most once or twice.\n"
    "3. Once a task has been completed, no further action is required for that task.\n"
    "4. If you're unsure about what the next step is, or if no action is needed, choose `FINISH` to indicate that the task is complete."
    "\n\n"
    "**Important Notes:**\n"
    "- Never enter an infinite loop. After completing a task or query, transition to the next step or finalize the process with `FINISH`."
    "- Always ensure that each task has a clear transition towards completion. Once a task is completed, the next step should be to finalize or delegate a new task."
    "- Keep the workflow clear, and avoid excessive iterations over the same task."
)


    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options] # * unpacks list options

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # output should be in the structure dictated by the Router class
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


# Research Agent Nodes

search_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=(
        "Do not write blogs or documents "
        "Don't ask follow-up questions."
    ),)


def search_node(state: State) -> Command[Literal["research_supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="research_supervisor",
    )


web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])


def web_scraper_node(state: State) -> Command[Literal["research_supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="research_supervisor",
    )


rag_agent = create_react_agent(
    llm,
    tools=[retrieve_recent_events_rag],  # Ensure it's added here
    prompt=(
        "Just return the retrieved documents, don't write blogs or documents"
        # "If query has `top AI agents` in it just return the retrieved documents, don't write blogs or documents yourself."
        # "If it does not have `top AI agents` in it, don't respond with anything."
        "Don't ask follow-up questions."
    ),
)

def rag_agent_node(state: State) -> Command[Literal["research_supervisor"]]:
    query = state["messages"][-1].content
    result = rag_agent.invoke(state)

    response = result["messages"][-1].content
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="rag_agent")
            ]
        },
        goto="research_supervisor",
    )

# ========================== Build research graph ===================================

research_supervisor_node = make_research_supervisor_node(llm, ["search", "web_scraper", "rag_agent"])
research_builder = StateGraph(State)
research_builder.add_node("research_supervisor", research_supervisor_node)
research_builder.add_node("rag_agent", rag_agent_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "research_supervisor")
research_graph = research_builder.compile()

# ========================== Document writer Team ===================================

# Writing Supervisor
def make_writing_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
    "You are a writing supervisor managing a structured workflow involving {members}. "
    "Your job is to efficiently coordinate tasks based on the user's request."
    "\n\n"
    "**Task Progression Rules:**\n"
    "1. If the query asks to write document or blog, limit your response to at most two calls to `doc_writer`\n"
    "2. If the query asks to write notes, limit your response to at most one or two calls to `note_taker`\n"
    "3. If the query asks to generate chart, limit your response to at most one or two calls to `chart_generator`\n"
    "4. Once a task has been completed, no further action is required for that task.\n"
    "5. If you're unsure about what the next step is, or if no action is needed, choose `FINISH` to indicate that the task is complete."
    "\n\n"
    "**Important Notes:**\n"
    "- Never enter an infinite loop. After completing a task or query, transition to the next step or finalize the process with `FINISH`."
    "- Always ensure that each task has a clear transition towards completion. Once a task is completed, the next step should be to finalize or delegate a new task."
    "- Keep the workflow clear, and avoid excessive iterations over the same task."
)

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options] # * unpacks list options

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        # output should be in the structure dictated by the Router class
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})


    return supervisor_node

# Writer Agent Nodes

doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines and save it. "
        "If user did not provide a file name, don't assume a file name. " # and don't create a file. "
        "Don't ask follow-up questions."
    ),
)


def doc_writing_node(state: State) -> Command[Literal["writing_supervisor"]]:
    result = doc_writer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="writing_supervisor",
    )


note_taking_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the document writer. "
        "Don't ask follow-up questions."
    ),
)


def note_taking_node(state: State) -> Command[Literal["writing_supervisor"]]:
    result = note_taking_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="writing_supervisor",
    )


chart_generating_agent = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)


def chart_generating_node(state: State) -> Command[Literal["writing_supervisor"]]:
    result = chart_generating_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="writing_supervisor",
    )


doc_writing_supervisor_node = make_writing_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)

# ========================== Build writing team graph ===================================

paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("writing_supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)

paper_writing_builder.add_edge(START, "writing_supervisor")
paper_writing_graph = paper_writing_builder.compile()

# ========================== Define components for supergraph ===================================


teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

def call_research_team(state: State) -> Command[Literal["supervisor"]]:
    response_messages = []

    # Use streaming instead of invoke()
    for s in research_graph.stream(state):
        print(s)  # Debugging output to see if it generates anything
        response_messages.append(HumanMessage(content=s["messages"][-1].content, name="research_team"))

    if not response_messages:
        return Command(update={}, goto="supervisor")  # Avoid infinite loops

    return Command(update={"messages": response_messages}, goto="supervisor")


def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
    response_messages = []

    # Use streaming instead of invoke()
    for s in paper_writing_graph.stream(state):
        print(s)  # Debugging output to see if it generates anything
        response_messages.append(HumanMessage(content=s["messages"][-1].content, name="writing_team"))

    if not response_messages:
        return Command(update={}, goto="supervisor")  # Avoid infinite loops

    return Command(update={"messages": response_messages}, goto="supervisor")

# ========================== Build a supergraph ===================================

super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

# ========================= Query ===============================

import sys

input_query = sys.argv[1]
print("Received query:", input_query)

@workflow(name="AgenticRAG")
def main():
    for s in super_graph.stream(
        {
            "messages": [
                ("user", input_query)
            ],
        },
        {"recursion_limit": 50},
    ):
        print(s)
        print("---")

if __name__ == "__main__":
    main()

# ========================== To Test milvus if running ===================================
# from db_milvus_scratch import run_search
# print(run_search("What is top-1 AI agent?"))


