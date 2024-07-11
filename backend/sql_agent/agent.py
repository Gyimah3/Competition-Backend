from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Any
import logging

from utils import get_chat_openai, get_chat_gemini
from tools.functions_tools import sql_agent_tools
from database.sql_db_langchain import db
from .agent_constants import CUSTOM_SUFFIX

logger = logging.getLogger(__name__)

def get_sql_toolkit(tool_llm_name: str):
    if tool_llm_name == "gpt-4o":
        llm_tool = get_chat_openai(model_name=tool_llm_name)
    elif tool_llm_name == "gemini-pro":
        llm_tool = get_chat_gemini(model_name=tool_llm_name)
    else:
        raise ValueError(f"Unsupported tool LLM: {tool_llm_name}")
    
    return SQLDatabaseToolkit(db=db, llm=llm_tool)

def get_agent_llm(agent_llm_name: str):
    if agent_llm_name == "gpt-4o":
        return get_chat_openai(model_name=agent_llm_name)
    elif agent_llm_name == "gemini-pro":
        return get_chat_gemini(model_name=agent_llm_name)
    else:
        raise ValueError(f"Unsupported agent LLM: {agent_llm_name}")

def create_retail_agent(
    tool_llm_name: str = "gpt-4o",
    agent_llm_name: str = "gpt-4o",
    db_session: Any = None
):
    agent_tools = sql_agent_tools(db_session)
    #retriever_tools = get_retriever_tool()
    llm_agent = get_agent_llm(agent_llm_name)
    toolkit = get_sql_toolkit(tool_llm_name)
    memory = ConversationBufferMemory(memory_key="history", input_key="input")

    return create_sql_agent(
        llm=llm_agent,
        toolkit=toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        input_variables=["input", "agent_scratchpad", "history"],
        suffix=CUSTOM_SUFFIX,
        agent_executor_kwargs={"memory": memory, "handle_parsing_errors": True},
        extra_tools=agent_tools,
        verbose=True,
    )

def run_agent(agent, input_text: str, chat_history: list):
    with get_openai_callback() as cb:
        response = agent.invoke(input=input_text, chat_history=chat_history)
    
    return response, cb.total_tokens, cb.total_cost
