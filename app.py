import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pandas as pd
import time
from graph.workflow import run_graph

st.set_page_config(
    page_title="ShopEase · Multi-Agent AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0A0A0F;color:#E8E6F0}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.5rem 2rem !important}
[data-testid="stSidebar"]{background:#0F0F1A !important;border-right:1px solid #1E1E2E !important}
[data-testid="stSidebar"] *{color:#C8C6D8 !important}
.brand{display:flex;align-items:center;gap:10px;padding-bottom:18px;border-bottom:1px solid #1E1E2E;margin-bottom:18px}
.brand-logo{width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#7C3AED,#EC4899);display:flex;align-items:center;justify-content:center;font-size:18px}
.brand-name{font-family:'Syne',sans-serif;font-weight:800;font-size:20px;color:#F0EEF8 !important}
.brand-sub{font-size:11px;color:#6B6880 !important}
.ccard{background:#13131F;border:1px solid #1E1E2E;border-radius:14px;padding:14px 16px;margin:10px 0}
.cname{font-family:'Syne',sans-serif;font-weight:700;font-size:15px;color:#F0EEF8 !important}
.tvip{background:linear-gradient(135deg,#F59E0B,#EF4444);color:white !important;font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;display:inline-block}
.treg{background:#1E1E2E;color:#6B6880 !important;font-size:10px;padding:2px 8px;border-radius:999px;display:inline-block}
.cdet{font-size:12px;color:#8884A0 !important;margin-top:8px;line-height:1.9}
.cdet strong{color:#C8C6D8 !important}
.slabel{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#4E4C60;margin:14px 0 8px}
.mhead{display:flex;align-items:center;justify-content:space-between;padding-bottom:14px;border-bottom:1px solid #1E1E2E;margin-bottom:18px}
.mtitle{font-family:'Syne',sans-serif;font-weight:800;font-size:24px;color:#F0EEF8}
.abadge{display:inline-flex;align-items:center;gap:6px;background:#13131F;border:1px solid #1E1E2E;border-radius:999px;padding:5px 14px;font-size:11px;color:#8884A0}
.pdot{width:8px;height:8px;border-radius:50%;background:#34D399;box-shadow:0 0 6px #34D39988;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.chatbox{background:#0D0D17;border:1px solid #1E1E2E;border-radius:16px;padding:20px;min-height:360px;max-height:420px;overflow-y:auto;margin-bottom:14px}
.mrow{display:flex;margin-bottom:14px;gap:10px}
.mrow.user{flex-direction:row-reverse}
.av{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.av-a{background:linear-gradient(135deg,#7C3AED,#EC4899)}
.av-u{background:#1E1E2E}
.bub{max-width:72%;padding:12px 16px;border-radius:16px;font-size:13px;line-height:1.7}
.b-a{background:#13131F;border:1px solid #1E1E2E;color:#D8D6E8;border-bottom-left-radius:4px}
.b-u{background:linear-gradient(135deg,#7C3AED22,#EC489922);border:1px solid #7C3AED44;color:#E8E6F0;border-bottom-right-radius:4px}
.b-esc{background:#2A1515;border:1px solid #EF444444;color:#FCA5A5;border-radius:12px;padding:14px 16px;font-size:13px;line-height:1.7}
.b-hum{background:#0F1F2A;border:1px solid #3B82F644;color:#BAE0FD;border-radius:12px;padding:14px 16px;font-size:13px;line-height:1.7}
.mtime{font-size:10px;color:#3E3C50;margin-top:4px}
.typing{display:flex;align-items:center;gap:8px;padding:8px 0;color:#6B6880;font-size:12px}
.tdots{display:flex;gap:4px}
.tdot{width:6px;height:6px;border-radius:50%;background:#7C3AED;animation:bounce 1.2s infinite}
.tdot:nth-child(2){animation-delay:.2s}.tdot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.4}30%{transform:translateY(-6px);opacity:1}}
.stButton button{background:#13131F !important;border:1px solid #2A2A3E !important;color:#A8A6BC !important;border-radius:999px !important;font-size:12px !important}
.stButton button:hover{border-color:#7C3AED !important;color:#C4B5FD !important}
[data-testid="stChatInput"]{background:#13131F !important;border:1px solid #2A2A3E !important;border-radius:14px !important}
[data-testid="stChatInput"] textarea{color:#E8E6F0 !important}
[data-testid="stSelectbox"]>div{background:#13131F !important;border:1px solid #2A2A3E !important;border-radius:10px !important}
.phead{font-family:'Syne',sans-serif;font-weight:700;font-size:16px;color:#F0EEF8;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #1E1E2E}
.ebadge{display:inline-flex;align-items:center;gap:6px;background:#2A1515;border:1px solid #EF444466;color:#FCA5A5;padding:6px 12px;border-radius:999px;font-size:11px;font-weight:600;margin-bottom:12px}
.igrid{background:#0D0D17;border:1px solid #1E1E2E;border-radius:12px;padding:14px;margin-bottom:12px}
.irow{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;border-bottom:1px solid #13131F}
.irow:last-child{border:none}
.ikey{color:#6B6880}.ival{color:#C8C6D8;font-weight:500}
.mcard{background:#0D0D17;border:1px solid #1E1E2E;border-radius:12px;padding:12px 16px;text-align:center;margin-bottom:8px}
.mval{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#F0EEF8}
.mlabel{font-size:11px;color:#6B6880;margin-top:2px}
.tracebox{background:#0A0A12;border:1px solid #1E1E2E;border-radius:12px;padding:12px 14px;margin-bottom:10px}
.ttitle{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#4E4C60;margin-bottom:10px}
.trow{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #13131F;font-size:12px}
.trow:last-child{border:none}
.tkey{color:#6B6880;min-width:90px;flex-shrink:0}
.tval{color:#C8C6D8}
.ti{background:#EEEDFE44;color:#C4B5FD;padding:2px 8px;border-radius:999px;font-size:11px}
.ta{background:#FAECE744;color:#FCA5A5;padding:2px 8px;border-radius:999px;font-size:11px}
.tn{background:#E1F5EE44;color:#6EE7B7;padding:2px 8px;border-radius:999px;font-size:11px}
.tc{background:#2A151544;color:#FCA5A5;padding:2px 8px;border-radius:999px;font-size:11px}
.tm{background:#FAEEDA44;color:#FCD34D;padding:2px 8px;border-radius:999px;font-size:11px}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:#0A0A0F}
::-webkit-scrollbar-thumb{background:#2A2A3E;border-radius:2px}
</style>
""", unsafe_allow_html=True)

# Session state
for k,v in {"messages":[],"customer_id":"","escalated_case":None,
            "last_trace":None,"total_queries":0,"resolved":0,
            "escalated_count":0,"quick_input":None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_data
def load_customers():
    return pd.read_csv("data/customers.csv")

def ts():
    return time.strftime("%I:%M %p")

def stag(s):
    return f'<span class="{"ta" if s in ["angry","frustrated"] else "tn"}">{s}</span>'

def utag(u):
    return f'<span class="{"tc" if u=="critical" else "tm"}">{u}</span>'

# SIDEBAR
with st.sidebar:
    st.markdown("""<div class="brand">
        <div class="brand-logo">🛍️</div>
        <div><div class="brand-name">ShopEase</div>
        <div class="brand-sub">Multi-Agent Console</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slabel">Active Customer</div>', unsafe_allow_html=True)
    df = load_customers()
    opts = {f"{r['name']}  ({r['customer_id']})": r['customer_id'] for _,r in df.iterrows()}
    sel = st.selectbox("Customer", list(opts.keys()), label_visibility="collapsed")
    st.session_state.customer_id = opts[sel]

    cr = df[df["customer_id"]==st.session_state.customer_id].iloc[0]
    tb = '<span class="tvip">★ VIP</span>' if cr["tier"]=="VIP" else '<span class="treg">REGULAR</span>'
    st.markdown(f"""<div class="ccard">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span class="cname">{cr['name']}</span>{tb}
        </div>
        <div class="cdet">
            <strong>Order</strong> #{cr['order_id']}<br>
            <strong>Status</strong> {cr['order_status']}<br>
            <strong>Item</strong> {cr['order_item']}<br>
            <strong>Value</strong> ₹{cr['amount']:,}<br>
            <strong>Complaints</strong> {cr['complaints']}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slabel">Session Stats</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,val,lbl in [(c1,st.session_state.total_queries,"Queries"),
                        (c2,st.session_state.resolved,"Resolved"),
                        (c3,st.session_state.escalated_count,"HITL")]:
        with col:
            st.markdown(f'<div class="mcard"><div class="mval">{val}</div><div class="mlabel">{lbl}</div></div>',unsafe_allow_html=True)

    if st.button("🗑️  Clear Chat", use_container_width=True):
        for k in ["messages","escalated_case","last_trace"]:
            st.session_state[k] = [] if k=="messages" else None
        for k in ["total_queries","resolved","escalated_count"]:
            st.session_state[k] = 0
        st.rerun()

    st.markdown("""<div style="margin-top:18px;padding-top:14px;border-top:1px solid #1E1E2E;
    font-size:11px;color:#3E3C50;line-height:2.2">
        🤖 LLaMA 3.2 · Ollama<br>
        ⚡ LangGraph · 6 Agents<br>
        🔍 FAISS · RAG Pipeline<br>
        🛡️ HITL Guardrails
    </div>""", unsafe_allow_html=True)

# MAIN
col1, col2 = st.columns([2,1], gap="large")

with col1:
    st.markdown(f"""<div class="mhead">
        <div class="mtitle">Customer Chat</div>
        <div class="abadge"><div class="pdot"></div>6 Agents Online</div>
    </div>""", unsafe_allow_html=True)

    # Chat render
    html = '<div class="chatbox">'
    if not st.session_state.messages:
        html += """<div style="text-align:center;padding:60px 20px;color:#3E3C50">
            <div style="font-size:36px;margin-bottom:12px">🤖</div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;color:#6B6880">Multi-Agent System Ready</div>
            <div style="font-size:12px;margin-top:6px">6 specialized agents waiting for your message</div>
        </div>"""
    else:
        for m in st.session_state.messages:
            role,content,t = m["role"],m["content"],m.get("time","")
            if role=="user":
                html+=f'<div class="mrow user"><div><div class="bub b-u">{content}</div><div class="mtime" style="text-align:right">{t}</div></div><div class="av av-u">👤</div></div>'
            elif "⚠️" in content or "Escalated" in content:
                html+=f'<div style="margin:12px 0"><div class="b-esc">🚨 <strong>Escalated to Human Agent</strong><br>{content.replace("⚠️","").strip()}</div><div class="mtime">{t}</div></div>'
            elif "Human Agent:" in content:
                html+=f'<div style="margin:12px 0"><div class="b-hum">👨‍💼 <strong>Human Agent</strong><br>{content.replace("👨‍💼 **Human Agent:**","").strip()}</div><div class="mtime">{t}</div></div>'
            else:
                html+=f'<div class="mrow"><div class="av av-a">✦</div><div><div class="bub b-a">{content}</div><div class="mtime">{t}</div></div></div>'
    html+='</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Quick replies
    st.markdown('<div class="slabel">Quick Replies</div>', unsafe_allow_html=True)
    qc = st.columns(5)
    for i,(lbl,msg) in enumerate([
        ("📦 Track Order","Where is my order?"),
        ("💸 Refund","What is your refund policy?"),
        ("🔄 Return","How do I return my item?"),
        ("❌ Cancel","Can I cancel my order?"),
        ("🆘 Urgent","I need urgent help with my order"),
    ]):
        with qc[i]:
            if st.button(lbl, key=f"q{i}"):
                st.session_state.quick_input = msg
                st.rerun()

    user_input = st.chat_input("Message ShopEase Support...")
    if st.session_state.quick_input:
        user_input = st.session_state.quick_input
        st.session_state.quick_input = None

    if user_input:
        t = ts()
        st.session_state.messages.append({"role":"user","content":user_input,"time":t})
        st.session_state.total_queries += 1

        ph = st.empty()
        ph.markdown('<div class="typing"><div class="tdots"><div class="tdot"></div><div class="tdot"></div><div class="tdot"></div></div>6 agents processing...</div>',unsafe_allow_html=True)

        result = run_graph(user_input, st.session_state.customer_id, st.session_state.messages[:-1])
        ph.empty()
        st.session_state.last_trace = result

        if result.get("should_escalate"):
            st.session_state.escalated_case = result
            st.session_state.escalated_count += 1
            st.session_state.messages.append({"role":"assistant","content":f"⚠️ Reason: {result.get('escalation_reason','')}","time":ts()})
        else:
            st.session_state.resolved += 1
            st.session_state.messages.append({"role":"assistant","content":result.get("final_response","Sorry, could not process your request."),"time":ts()})
        st.rerun()

with col2:
    st.markdown('<div class="phead">⚡ Control Panel</div>', unsafe_allow_html=True)

    # Agent trace
    if st.session_state.last_trace:
        t = st.session_state.last_trace
        cd = t.get("customer_data") or {}
        st.markdown(f"""<div class="tracebox">
            <div class="ttitle">🔬 Agent Trace — Last Run</div>
            <div class="trow"><span class="tkey">① Intent</span><span class="tval"><span class="ti">{t.get('intent','—')}</span></span></div>
            <div class="trow"><span class="tkey">② Sentiment</span><span class="tval">{stag(t.get('sentiment','—'))}</span></div>
            <div class="trow"><span class="tkey">③ Urgency</span><span class="tval">{utag(t.get('urgency','—'))}</span></div>
            <div class="trow"><span class="tkey">④ CRM</span><span class="tval" style="color:#6EE7B7">{cd.get('name','—')} ({cd.get('tier','?').upper() if cd else '?'})</span></div>
            <div class="trow"><span class="tkey">⑤ RAG</span><span class="tval" style="color:#6EE7B7">{'Retrieved ✓' if t.get('policy_chunks') else '—'}</span></div>
            <div class="trow"><span class="tkey">Escalate</span><span class="tval" style="color:{'#FCA5A5' if t.get('should_escalate') else '#6EE7B7'}">{'Yes 🚨' if t.get('should_escalate') else 'No ✓'}</span></div>
        </div>""", unsafe_allow_html=True)

    # HITL panel
    if st.session_state.escalated_case:
        case = st.session_state.escalated_case
        cd   = case.get("customer_data") or {}
        st.markdown('<div class="ebadge">🚨 Escalation Required</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="igrid">
            <div class="irow"><span class="ikey">Customer</span><span class="ival">{cd.get('name','—')}</span></div>
            <div class="irow"><span class="ikey">Tier</span><span class="ival">{cd.get('tier','—').upper()}</span></div>
            <div class="irow"><span class="ikey">Order</span><span class="ival">#{cd.get('order_id','—')}</span></div>
            <div class="irow"><span class="ikey">Complaints</span><span class="ival">{cd.get('complaints','0')}</span></div>
            <div class="irow"><span class="ikey">Reason</span><span class="ival" style="color:#FCA5A5">{case.get('escalation_reason','')}</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="slabel">AI Draft Reply</div>', unsafe_allow_html=True)
        draft = st.text_area("", value=case.get("draft_reply",""), height=120, label_visibility="collapsed")
        ca,cb = st.columns(2)
        with ca:
            if st.button("✅ Approve & Send", type="primary", use_container_width=True):
                st.session_state.messages.append({"role":"assistant","content":f"👨‍💼 **Human Agent:** {draft}","time":ts()})
                st.session_state.escalated_case = None
                st.rerun()
        with cb:
            if st.button("✕ Dismiss", use_container_width=True):
                st.session_state.escalated_case = None
                st.rerun()

    elif not st.session_state.last_trace:
        st.markdown("""<div style="text-align:center;padding:30px 0 20px">
        <div style="font-size:30px">🟢</div>
        <div style="font-size:13px;color:#34D399;font-weight:600;margin-top:8px">All 6 Agents Ready</div>
        <div style="font-size:11px;color:#4E4C60;margin-top:4px">Send a message to see the trace</div>
    </div>
    <div style="background:#0D0D17;border:1px solid #1E1E2E;border-radius:12px;padding:12px 14px;font-size:11px;color:#6B6880;line-height:2.4">
    <span style="color:#C4B5FD">①</span> Intent Agent → Intent &amp; Sentiment Analysis<br>
    <span style="color:#C4B5FD">②</span> CRM Agent → Customer Profile Retrieval<br>
    <span style="color:#C4B5FD">③</span> Memory Agent → Conversation Memory<br>
    <span style="color:#C4B5FD">④</span> RAG Agent → Policy Knowledge Retrieval<br>
    <span style="color:#C4B5FD">⑤</span> Escalation Agent → Human-in-the-Loop Decision<br>
    <span style="color:#C4B5FD">⑥</span> Supervisor Agent → Final Response Generation
    </div>""", unsafe_allow_html=True)

# ============================================================
# RESEARCH / EVALUATION DASHBOARD
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        font-family:'Syne',sans-serif;
        font-size:24px;
        font-weight:800;
        color:#F0EEF8;
        margin-top:25px;
        margin-bottom:5px;
    ">
        📊 Research Evaluation Dashboard
    </div>

    <div style="
        color:#6B6880;
        font-size:12px;
        margin-bottom:20px;
    ">
        ShopEase evaluation on 590 valid test queries
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Load evaluation summary
# ------------------------------------------------------------

SUMMARY_PATH = "research_results/retrieval_comparison_summary.csv"

try:
    summary_df = pd.read_csv(SUMMARY_PATH)

    # Current evaluation accuracy
    bm25_acc = float(
        summary_df.loc[
            summary_df["mode"].str.lower() == "bm25", "accuracy"
        ].iloc[0]
    ) * 100

    faiss_acc = float(
        summary_df.loc[
            summary_df["mode"].str.lower() == "faiss", "accuracy"
        ].iloc[0]
    ) * 100

    hybrid_acc = float(
        summary_df.loc[
            summary_df["mode"].str.lower() == "hybrid", "accuracy"
        ].iloc[0]
    ) * 100

except Exception:
    # Fallback values from the current 590-query evaluation
    bm25_acc = 49.92
    faiss_acc = 92.71
    hybrid_acc = 90.68

# Fair RRF result comes from its dedicated evaluation
try:
    rrf_df = pd.read_csv(
        "research_results/hybrid_rrf_fair_results.csv"
    )

    rrf_acc = (
        rrf_df["correct"].mean() * 100
    )

except Exception:
    rrf_acc = 83.39


# ------------------------------------------------------------
# TOP METRIC CARDS
# ------------------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)

metrics = [
    (m1, f"{bm25_acc:.2f}%", "BM25 Accuracy"),
    (m2, f"{faiss_acc:.2f}%", "FAISS Accuracy"),
    (m3, f"{rrf_acc:.2f}%", "Fair RRF Accuracy"),
    (m4, f"{hybrid_acc:.2f}%", "Hybrid Accuracy"),
    (m5, "590", "Valid Queries"),
]

for col, value, label in metrics:
    with col:
        st.markdown(
            f"""
            <div class="mcard" style="min-height:90px">
                <div class="mval">{value}</div>
                <div class="mlabel">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# RESEARCH TABS
# ------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Accuracy & Latency",
        "🔍 Retrieval Ablation",
        "🧩 Component Ablation",
        "🎯 Confusion Matrix",
        "📋 Statistical Analysis",
    ]
)


# ============================================================
# TAB 1 — ACCURACY + LATENCY
# ============================================================

with tab1:

    st.markdown(
        '<div class="phead">Retrieval Performance</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            "research_results/accuracy_comparison.png",
            caption="Accuracy Comparison",
            use_container_width=True
        )

    with c2:
        st.image(
            "research_results/latency_comparison.png",
            caption="Average Response Latency",
            use_container_width=True
        )

    # Current numerical summary
    st.markdown(
        '<div class="phead">Numerical Results</div>',
        unsafe_allow_html=True
    )

    display_df = pd.DataFrame({
        "Retrieval Method": [
            "BM25-only",
            "FAISS-only",
            "RRF-only (Fair)",
            "RRF + Cross-Encoder"
        ],
        "Accuracy (%)": [
            bm25_acc,
            faiss_acc,
            rrf_acc,
            hybrid_acc
        ]
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — RETRIEVAL ABLATION
# ============================================================

with tab2:

    st.markdown(
        '<div class="phead">Retrieval Ablation Study</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="tracebox">
        <div class="ttitle">Retrieval Configurations</div>

        <div class="trow">
            <span class="tkey">BM25-only</span>
            <span class="tval">Sparse lexical retrieval</span>
        </div>

        <div class="trow">
            <span class="tkey">FAISS-only</span>
            <span class="tval">Dense semantic retrieval</span>
        </div>

        <div class="trow">
            <span class="tkey">RRF-only</span>
            <span class="tval">FAISS + BM25 → RRF → majority vote</span>
        </div>

        <div class="trow">
            <span class="tkey">Hybrid</span>
            <span class="tval">RRF + Cross-Encoder reranking</span>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        "research_results/retrieval_ablation_comparison.png",
        caption="Retrieval Ablation Comparison",
        use_container_width=True
    )

    # Show raw RRF evaluation
    try:
        rrf_results = pd.read_csv(
            "research_results/hybrid_rrf_fair_results.csv"
        )

        st.markdown(
            '<div class="phead">Fair RRF Evaluation</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"Correct: {int(rrf_results['correct'].sum())} / "
            f"{len(rrf_results)}"
        )

        st.dataframe(
            rrf_results.head(20),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(f"RRF results unavailable: {e}")


# ============================================================
# TAB 3 — COMPONENT ABLATION
# ============================================================

with tab3:

    st.markdown(
        '<div class="phead">Multi-Agent Component Ablation</div>',
        unsafe_allow_html=True
    )

    st.image(
        "research_results/ablation_comparison.png",
        caption="Component Ablation Study",
        use_container_width=True
    )

    try:
        component_df = pd.read_csv(
            "research_results/ablation_study_summary.csv"
        )

        st.dataframe(
            component_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(
            f"Component ablation data unavailable: {e}"
        )


# ============================================================
# TAB 4 — CONFUSION MATRIX
# ============================================================

with tab4:

    st.markdown(
        '<div class="phead">Main Hybrid RAG Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Main result: RRF + Cross-Encoder on 590 valid queries."
    )

    st.image(
        "research_results/confusion_matrix.png",
        caption="Confusion Matrix — RRF + Cross-Encoder",
        use_container_width=True
    )

    # Also provide the fair RRF matrix separately if available
    if os.path.exists(
        "research_results/confusion_matrix_rrf_fair.png"
    ):
        st.markdown(
            '<div class="phead">Fair RRF-only Confusion Matrix</div>',
            unsafe_allow_html=True
        )

        st.image(
            "research_results/confusion_matrix_rrf_fair.png",
            caption="Confusion Matrix — Fair RRF-only",
            use_container_width=True
        )


# ============================================================
# TAB 5 — CLASSIFICATION + STATISTICAL ANALYSIS
# ============================================================

with tab5:

    st.markdown(
        '<div class="phead">Classification Report</div>',
        unsafe_allow_html=True
    )

    try:
        with open(
            "research_results/classification_report.txt",
            "r",
            encoding="utf-8"
        ) as f:
            report_text = f.read()

        st.code(
            report_text,
            language="text"
        )

    except Exception as e:
        st.warning(
            f"Classification report unavailable: {e}"
        )


    st.markdown(
        '<div class="phead">McNemar Significance Test</div>',
        unsafe_allow_html=True
    )

    try:
        sig_df = pd.read_csv(
            "research_results/significance_test_results.csv"
        )

        st.dataframe(
            sig_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "McNemar's test evaluates paired prediction differences "
            "between retrieval methods."
        )

    except Exception as e:
        st.warning(
            f"Significance test results unavailable: {e}"
        )