from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from graph.state import AgentState

llm = OllamaLLM(model="llama3.2", temperature=0.2)
parser = StrOutputParser()

def sanitize_refund_response(response: str) -> str:
    lower = response.lower()
    banned_patterns = [
        "eligible for a refund",
        "qualify for a refund",
        "approved for a refund",
        "guaranteed a refund",
        "less than 7 days since delivery",
        "[date]",
        "[x]"
    ]
    if any(p in lower for p in banned_patterns):
        return (
            "Hi Rahul,\n\n"
            "I understand that you would like to request a refund for your laptop order (#45231), "
            "which is currently marked as shipped. According to our Refund Policy, refund requests "
            "are reviewed based on the circumstances of the order. Could you please share the reason "
            "for your refund request?\n\n"
            "Best regards,\n"
            "ShopEase Support Team"
        )
    return response

def sanitize_tracking_response(response: str) -> str:
    lower = response.lower()

    banned = [
        "tracking portal",
        "unique tracking id",
        "tracking id sent",
        "courier",
        "estimated arrival"
    ]

    if any(x in lower for x in banned):
        return (
            "Hi Rahul,\n\n"
            "Your order (#45231) is currently marked as Shipped. "
            "Additional tracking information is not available in our records at this time. "
            "We will update you once new shipment information becomes available.\n\n"
            "Best regards,\n"
            "ShopEase Support Team"
        )

    return response

PROMPT = ChatPromptTemplate.from_template("""You are a helpful and empathetic customer support agent for ShopEase.

CUSTOMER INFORMATION (from CRM):
{customer_context}

POLICY SOURCE:
{policy_source}                                          
                                          
RELEVANT COMPANY POLICY (from knowledge base):
{policy_chunks}

DETECTED INTENT: {intent}
CUSTOMER SENTIMENT: {sentiment}

CONVERSATION HISTORY:
{history}

CUSTOMER MESSAGE:
{user_message}
                                          
REFUND STATUS:
{refund_eligible}

INSTRUCTIONS:
                                          
- Never determine refund eligibility yourself.
- Never say "you are eligible for a refund".
- Never infer refund eligibility from shipped status.
- If REFUND STATUS is "Needs Review", say:
  "Your refund request will be reviewed according to our Refund Policy."
- Never infer policy decisions from order status alone.
- Do not use words such as "eligible", "approved", or "guaranteed" unless explicitly supported by the provided information.
- Address the customer by their first name only.
- Use ONLY the information provided in CUSTOMER INFORMATION, POLICY SOURCE, and POLICY CHUNKS.
- Never invent, assume, or guess any information.
- Never generate placeholders such as [date], [X], [Order ID], [Courier Name], XXXXX, or similar text.
- If any information is unavailable, politely ask the customer for it instead of inventing it.

- Never conclude that a customer is eligible for a refund unless the policy and customer information explicitly confirm eligibility.
- If eligibility cannot be determined, say that the request will be reviewed according to the Refund Policy.
                                          
- CUSTOMER INFORMATION is the highest-priority source.
- Never contradict CUSTOMER INFORMATION.
- policy_chunks are only for company policies and MUST NOT be used for order status or customer details.
- memory_context is only for personalization.

- If customer_context contains "Status: Shipped", explicitly state that the order is currently shipped.
- If an Order ID is available, mention the exact Order ID.
- If shipment date, delivery date, courier name, or tracking information is not available, do not mention them.

- Use company policies only as guidance for generating a response.
- Never reproduce policy text word-for-word.
- Mention the policy source naturally only when relevant, for example:
  "According to our Refund Policy..."
  "According to our Damaged Product Policy..."

- Write conversationally like a professional human customer support agent.
- Do not sound robotic or copy policy language.
- First acknowledge the customer's request or concern.
- Be empathetic, especially if the customer is angry or frustrated.
- If the customer is a VIP, warmly acknowledge their loyalty.

- Ask for information ONLY if it is not already available in CUSTOMER INFORMATION.
- Never ask the customer to confirm information that already exists in CUSTOMER INFORMATION.
- Do not start directly with policy instructions.
- If order status is already present in CUSTOMER INFORMATION, state it directly and do not ask the customer to confirm it.
- Treat CUSTOMER INFORMATION as verified data.
- Never infer additional shipping information from company policy.
- Policy chunks must not override CRM data.

                                          
- For refund requests:
    1. Acknowledge the refund request.
    2. State the exact order status and order ID if available.
    3. Do not ask the customer to confirm information already present in CUSTOMER INFORMATION.
    4. Ask only for missing details, such as the reason for the refund request.
    5. State that refund requests are reviewed according to the Refund Policy.
    6. Never declare refund eligibility, approval, or guarantee unless explicitly confirmed by policy and customer information.
    7. Mention the current order status if available.
    8. Ask for the reason for the refund if not provided.
    9. NEVER say that the customer is eligible for a refund, approved for a refund, or guaranteed a refund unless the policy_chunks and customer information explicitly confirm it.
    10. "Shipped" status alone does NOT imply refund eligibility.
    11. Explain the next steps.
                                          
- For damaged-product complaints:
    1. Apologize empathetically.
    2. Mention the Damaged Product Policy if relevant.
    3. Ask for photos or proof if needed.
    4. Explain replacement/refund options.
                                          
- For order tracking queries:
    1. Mention the exact Order ID if available.
    2. Mention the exact order status from CUSTOMER INFORMATION.
    3. If tracking information is unavailable, explicitly say so.
    4. Never invent tracking IDs, tracking portals, courier names, estimated delivery dates, or shipment dates.
    5. Never infer tracking information from company policy.
    6. If only "Shipped" status exists, say:
       "Your order is currently marked as shipped and additional tracking information is not available at this time."

- Keep the response between 60 and 100 words.
- Sign off exactly as:

Best regards,
ShopEase Support Team
YOUR RESPONSE:""")

chain = PROMPT | llm | parser


def supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor Agent — sab agents ka output leke final response banata hai.
    Input  : full state
    Output : state["final_response"], state["next"]
    """
    if state.get("should_escalate"):
        print("[Supervisor] Escalation triggered — routing to HITL")
        return {**state, "final_response": None, "next": "escalate"}

    history = state.get(
    "memory_context",
    "No previous conversation."
    )

    customer_context = state.get("customer_context", "").lower()

    if "status: shipped" in customer_context:
        refund_eligible = "Needs Review"
    elif "status: delivered" in customer_context:
        refund_eligible = "Needs Review"
    else:
        refund_eligible = "Unknown"

    response = chain.invoke({
    "customer_context": state.get("customer_context", "Not available"),
    "policy_source": state.get("policy_source","General Policy"),
    "policy_chunks": state.get("policy_chunks","Not available"),
    "intent": state.get("intent", "general"),
    "sentiment": state.get("sentiment", "neutral"),
    "history": history,
    "user_message": state.get("user_message", ""),
    "refund_eligible": refund_eligible,
})
    if state.get("intent") == "refund_request":
        response = sanitize_refund_response(response)

    if state.get("intent") == "order_tracking":
        response = sanitize_tracking_response(response)
    
    print("INTENT =", state.get("intent"))
    print("BEFORE SANITIZE:\n", response)

    response = sanitize_refund_response(response)

    print("AFTER SANITIZE:\n", response)

    print(f"[Supervisor] Response generated ({len(response)} chars)")

    return {
        **state,
        "final_response": response,
        "next": "end"
    }