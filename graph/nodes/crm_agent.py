from graph.state import AgentState
from utils.timing import track_latency
from utils.db import get_customer_by_id


@track_latency("crm_agent")
def crm_agent(state: AgentState) -> AgentState:
    """
    CRM Agent — SQLite (data/shopease.db) se ek parameterized SQL query
    ke through customer data fetch karta hai.
    Input  : state["customer_id"]
    Output : state["customer_data"], state["customer_context"]
    """
    customer_id = state.get("customer_id", "").strip()

    try:
        customer_data = get_customer_by_id(customer_id)

        if not customer_data:
            context = "Customer not found in system."
        else:
            context = (
                f"Name: {customer_data.get('name')}\n"
                f"Tier: {customer_data.get('tier', 'regular').upper()}\n"
                f"Order ID: #{customer_data.get('order_id')}\n"
                f"Order Status: {customer_data.get('order_status')}\n"
                f"Item: {customer_data.get('order_item')}\n"
                f"Order Amount: Rs {customer_data.get('amount')}\n"
                f"Past Complaints: {customer_data.get('complaints', 0)}\n"
                f"Member Since: {customer_data.get('member_since')}"
            )
    except Exception as e:
        customer_data = {}
        context = f"CRM error: {str(e)}"

    print(f"[CRM Agent] Fetched: {customer_data.get('name', 'Unknown')} | Tier: {customer_data.get('tier', '?')}")

    return {
        **state,
        "customer_data":    customer_data,
        "customer_context": context,
    }
