from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache
import logging
from config import OPENAI_API_KEY, GOOGLE_API_KEY

@lru_cache(maxsize=2)
def get_chat_openai(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=3000,
        streaming=True,
        verbose=False,
    )

@lru_cache(maxsize=2)
def get_chat_gemini(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=model_name,
        temperature=0,
        max_output_tokens=3000,
    )


import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "retail_insights.log")

    # Create a custom logger
    logger = logging.getLogger("retail_insights")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logging()