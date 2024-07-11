chat_openai_kwargs = {
    "temperature": 0,
    "max_tokens": 3000,
    "streaming": True,
    "verbose": False,
}

langchain_chat_kwargs = {
    "streaming": True,
    "verbose": False,
}

# Add caching configuration
CACHE_TYPE = "SimpleCache"
CACHE_DEFAULT_TIMEOUT = 3000