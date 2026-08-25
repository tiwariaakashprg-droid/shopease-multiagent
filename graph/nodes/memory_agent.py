from graph.state import AgentState
from utils.timing import track_latency


@track_latency("memory_agent")
def memory_agent(state: AgentState) -> AgentState:
    """
    Builds conversation memory from previous chats.

    NOTE: runs in PARALLEL with crm_agent (see graph/workflow.py) - both are
    read-only w.r.t. the incoming query and write to disjoint state keys, so
    LangGraph executes them in the same superstep instead of sequentially.
    """

    history = state.get("chat_history", [])

    if not history:
        memory_context = "No previous conversation."
    else:
        memory = []

        for msg in history[-5:]:   # last 5 messages
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                memory.append(f"{role}: {content}")
            else:
                memory.append(str(msg))

        memory_context = (
             "Previous customer interactions:\n"
             + "\n".join(memory)
        )

    print("[Memory Agent] Memory loaded.")

    return {**state, "memory_context": memory_context}