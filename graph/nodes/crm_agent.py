import pandas as pd
from graph.state import AgentState


def crm_agent(state: AgentState) -> AgentState:
    """
    CRM Agent — customers.csv se customer data fetch karta hai.
    Input  : state["customer_id"]
    Output : state["customer_data"], state["customer_context"]
    """
    customer_id = state.get("customer_id", "").strip()

    try:
        df = pd.read_csv("data/customers.csv")
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
        row = df[df["customer_id"] == customer_id]

        if row.empty:
            customer_data = {}
            context = "Customer not found in system."
        else:
            customer_data = row.iloc[0].to_dict()
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
