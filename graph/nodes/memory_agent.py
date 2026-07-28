from graph.state import AgentState


def memory_agent(state: AgentState) -> AgentState:
    """
    Builds conversation memory from previous chats.
    """

    history = state.get("chat_history", [])

    if not history:
        state["memory_context"] = "No previous conversation."
    else:
        memory = []

        for msg in history[-5:]:   # last 5 messages
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                memory.append(f"{role}: {content}")
            else:
                memory.append(str(msg))

        state["memory_context"] = (
             "Previous customer interactions:\n"
             + "\n".join(memory)
        )

    print("[Memory Agent] Memory loaded.")

    return state