import config
from app.llm.claude_provider import ClaudeProvider


def get_llm_provider():
    if config.LLM_PROVIDER == "anthropic":
        return ClaudeProvider()

    raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")