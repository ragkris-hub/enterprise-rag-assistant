from langchain_anthropic import ChatAnthropic
import config


class ClaudeProvider:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            temperature=0
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content