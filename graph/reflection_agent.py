from graph.state import AgentState
from utils.timing import track_latency


@track_latency("reflection_agent")
def reflection_agent(state: AgentState) -> AgentState:
    print("[Reflection Agent] Validation skipped.")
    return state
