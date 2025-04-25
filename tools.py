import os
import uuid
import gradio as gr
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("tavily_api_key")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "CourseLanggraph"
os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- Import and Initialize LangChain tools ---
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from typing import List
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langchain.schema import HumanMessage

# Define LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

# Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
tavily = TavilySearchResults(max_results=1)

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
def wiki_fun(question: str) -> str:
    """Searches Wikipedia for the given question and returns detailed information."""
    return wiki_tool.invoke(question)

@tool
def arxiv_fun(question: str) -> str:
    """Searches arXiv for academic papers relevant to the given question and returns detailed information."""
    return arxiv_tool.invoke(question)

@tool
def tavily_fun(question: str) -> str:
    """Uses Tavily to perform a web search based on the question and returns summarized results."""
    return tavily.invoke(question)

tools = [wiki_fun, arxiv_fun, tavily_fun, scrape_webpages]
# tools = [wiki_tool,arxiv_tool,tavily]
llm_with_tools = llm.bind_tools(tools=tools)

# Chat history manager
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define LangGraph State and Graph
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph_builder = StateGraph(State)

def chatbot(state:State):
    return {"messages":llm_with_tools.invoke(state['messages'])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile(checkpointer=memory)

# --- Gradio Chat Interface ---

session_id = str(uuid.uuid4())  # Unique ID per Gradio user session
config = {"configurable": {"thread_id": session_id}}

def chat_fn(message, history):
    input_msg = HumanMessage(content=message)
    output = None
    for event in graph.stream({"messages": [input_msg]}, config, stream_mode="values"):
        output = event["messages"][-1].content
    history.append((message, output))
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LangGraph Chatbot with Tools")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here...")
    clear = gr.Button("Clear")

    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(share=True)
