from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve.pydantic_v1 import BaseModel, Field
import asyncio
from typing import Any, List
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import logging

from sql_agent.agent import create_retail_agent, run_agent
from database.sql_db_langchain import get_db
from config import TOOL_LLM_NAME, AGENT_LLM_NAME
from utils import setup_logging
from tools.functions_tools import create_chart_image

setup_logging()
logger = logging.getLogger("retail_insights")

class Input(BaseModel):
    input: str = Field(..., description="The retail insights query")
    tool_llm_name: str = Field(default=TOOL_LLM_NAME, description="LLM for SQL tools")
    agent_llm_name: str = Field(default=AGENT_LLM_NAME, description="LLM for the agent")
    chat_history: List[str] = Field(default=[], description="Chat history")

class Output(BaseModel):
    output: Any = Field(..., description="The response from the Retail Insights Chatbot")
    tokens_used: int = Field(..., description="Number of tokens used in the query")
    cost: float = Field(..., description="Cost of the query")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application is starting up...")
    yield
    # Shutdown
    logger.info("Application is shutting down...")

app = FastAPI(
    title="Retail Insights Chatbot",
    version="1.0",
    description="API for a Retail Insights Chatbot using LangChain's SQL agent",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "https://hackathon-37g0uk0y2-gideon-gyimahs-projects.vercel.app", 
    "https://hackathon-97icvvxb7-gideon-gyimahs-projects.vercel.app"# Replace with your actual frontend URL
    "http://localhost:3000",  # If you are testing locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def truncate_chat_history(history, max_tokens=1000):
    truncated_history = []
    current_tokens = 0
    for message in reversed(history):
        message_tokens = len(message.split())  # Simple approximation
        if current_tokens + message_tokens > max_tokens:
            break
        truncated_history.insert(0, message)
        current_tokens += message_tokens
    return truncated_history

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retail Insights Chatbot API"}

@app.post("/query", response_model=Output)
async def query_retail_insights(
    request: Request,
    input: Input,
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Received query: {input.input}")
        logger.info(f"Chat history: {input.chat_history}")

        truncated_history = truncate_chat_history(input.chat_history)
        agent = create_retail_agent(input.tool_llm_name, input.agent_llm_name, db)
        response, tokens, cost = await asyncio.to_thread(
            run_agent, agent, input.input, truncated_history
        )

        # Check if the response contains chart data
        if isinstance(response, str) and "{'columns':" in response:
            # Extract the chart data dictionary from the string
            start_index = response.index("{'columns':")
            end_index = response.index("}", start_index) + 1
            chart_data_str = response[start_index:end_index]
            
            # Convert the string to a dictionary
            import ast
            chart_data = ast.literal_eval(chart_data_str)
            
            # Generate the chart
            chart_image = create_chart_image(chart_data)
            
            # Replace the chart data in the response with the chart image
            response = response[:start_index] + chart_image + response[end_index:]

        return {"output": response, "tokens_used": tokens, "cost": cost}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
