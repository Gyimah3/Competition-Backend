import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
TOOL_LLM_NAME = os.getenv("TOOL_LLM_NAME")
AGENT_LLM_NAME = os.getenv("AGENT_LLM_NAME")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Database Configuration
DB_PATH = os.getenv("DB_PATH")

# Other Configuration
MAX_RETRIES = 2
TIMEOUT = 60