from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.3
)

def single_agent_response(user_message: str) -> str:

    prompt = f"""
You are a customer support chatbot.

Answer the customer query naturally.

Customer Query:
{user_message}

Response:
"""

    return llm.invoke(prompt)