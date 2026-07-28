from graph.state import AgentState
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.2
)

def reflection_agent(state: AgentState) -> AgentState:
    print("[Reflection Agent] Validation skipped.")
    return state