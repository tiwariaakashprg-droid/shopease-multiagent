import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pandas as pd
import time
from graph.workflow import run_graph
from utils.db import get_all_customers

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
    # Backed by SQLite now (utils/db.py) — runs a real SQL query instead
    # of reading the CSV straight into a DataFrame.
    return get_all_customers()

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
        ShopEase evaluation on 2,632 test queries
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# EVALUATION PATHS
# ============================================================

EVAL_DIR = "research_results/final_2632_evaluation"

RESULT_FILES = {
    "BM25-only":
        os.path.join(EVAL_DIR, "evaluation_results_bm25.csv"),

    "FAISS-only":
        os.path.join(EVAL_DIR, "evaluation_results_faiss.csv"),

    "Fair RRF":
        os.path.join(EVAL_DIR, "evaluation_results_fair_rrf.csv"),

    "Weighted RRF":
        os.path.join(EVAL_DIR, "evaluation_results_weighted_rrf.csv"),

    "RRF + Cross-Encoder":
        os.path.join(EVAL_DIR, "evaluation_results_hybrid.csv"),

    "Top-10 Hybrid":
        os.path.join(EVAL_DIR, "evaluation_results_top10_hybrid.csv"),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_evaluation_file(path):
    """
    Load an evaluation CSV and validate its structure.
    """

    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)

        required_columns = [
            "query",
            "expected",
            "predicted",
            "correct"
        ]

        for column in required_columns:
            if column not in df.columns:
                return None

        return df

    except Exception:
        return None


def calculate_accuracy(df):
    """
    Calculate accuracy directly from the evaluation CSV.

    correct = 1 -> correct prediction
    correct = 0 -> incorrect prediction
    """

    if df is None or len(df) == 0:
        return 0.0

    return df["correct"].astype(int).mean() * 100


def calculate_correct(df):
    """
    Calculate number of correct predictions.
    """

    if df is None:
        return 0

    return int(df["correct"].astype(int).sum())


def calculate_latency(df):
    """
    Calculate average latency if latency column exists.
    """

    if df is None:
        return None

    if "latency" not in df.columns:
        return None

    try:
        return float(
            pd.to_numeric(
                df["latency"],
                errors="coerce"
            ).dropna().mean()
        )

    except Exception:
        return None


# ============================================================
# LOAD ALL EVALUATION RESULTS
# ============================================================

evaluation_data = {}

for method, path in RESULT_FILES.items():

    df = load_evaluation_file(path)

    if df is not None:
        evaluation_data[method] = df


# ============================================================
# GLOBAL QUERY COUNT
# ============================================================

query_counts = []

for method, df in evaluation_data.items():
    query_counts.append(len(df))

if query_counts:
    TOTAL_QUERIES = max(query_counts)
else:
    TOTAL_QUERIES = 2632


# ============================================================
# CALCULATE CURRENT RESULTS
# ============================================================

results_summary = []

for method, df in evaluation_data.items():

    correct = calculate_correct(df)
    accuracy = calculate_accuracy(df)
    latency = calculate_latency(df)

    results_summary.append(
        {
            "Method": method,
            "Queries": len(df),
            "Correct": correct,
            "Accuracy (%)": accuracy,
            "Average Latency (s)": latency
        }
    )


results_df = pd.DataFrame(results_summary)


# ============================================================
# TOP METRIC CARDS
# ============================================================

best_method = None
best_accuracy = 0.0

if not results_df.empty:

    best_row = results_df.loc[
        results_df["Accuracy (%)"].idxmax()
    ]

    best_method = best_row["Method"]
    best_accuracy = float(
        best_row["Accuracy (%)"]
    )


m1, m2, m3, m4, m5 = st.columns(5)


# BM25
bm25_value = "—"

if "BM25-only" in evaluation_data:
    bm25_value = (
        f"{calculate_accuracy(evaluation_data['BM25-only']):.2f}%"
    )


# FAISS
faiss_value = "—"

if "FAISS-only" in evaluation_data:
    faiss_value = (
        f"{calculate_accuracy(evaluation_data['FAISS-only']):.2f}%"
    )


# Fair RRF
fair_rrf_value = "—"

if "Fair RRF" in evaluation_data:
    fair_rrf_value = (
        f"{calculate_accuracy(evaluation_data['Fair RRF']):.2f}%"
    )


# Weighted RRF
weighted_rrf_value = "—"

if "Weighted RRF" in evaluation_data:
    weighted_rrf_value = (
        f"{calculate_accuracy(evaluation_data['Weighted RRF']):.2f}%"
    )


# Main Hybrid
hybrid_value = "—"

if "RRF + Cross-Encoder" in evaluation_data:
    hybrid_value = (
        f"{calculate_accuracy(evaluation_data['RRF + Cross-Encoder']):.2f}%"
    )


metrics = [
    (m1, bm25_value, "BM25 Accuracy"),
    (m2, faiss_value, "FAISS Accuracy"),
    (m3, fair_rrf_value, "Fair RRF Accuracy"),
    (m4, weighted_rrf_value, "Weighted RRF Accuracy"),
    (m5, f"{TOTAL_QUERIES:,}", "Evaluation Queries"),
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


# ============================================================
# BEST RESULT HIGHLIGHT
# ============================================================

if best_method is not None:

    best_correct = int(best_row["Correct"])

    st.markdown(
        f"""<div class="tracebox" style="margin-top:8px">
<div class="ttitle">🏆 BEST ACCURACY</div>

<div class="trow">
<span class="tkey">Configuration</span>
<span class="tval"><span class="tn">{best_method}</span></span>
</div>

<div class="trow">
<span class="tkey">Accuracy</span>
<span class="tval">{best_accuracy:.2f}%</span>
</div>

<div class="trow">
<span class="tkey">Correct</span>
<span class="tval">{best_correct:,} / {TOTAL_QUERIES:,}</span>
</div>

</div>""",
        unsafe_allow_html=True
    )

# ============================================================
# RESEARCH TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Accuracy & Latency",
        "🔍 Retrieval Ablation",
        "🧩 Component Ablation",
        "🎯 Confusion Matrix",
        "📋 Classification Reports",
        "📊 Statistical Analysis",
    ]
)


# ============================================================
# TAB 1
# ACCURACY + LATENCY
# ============================================================

with tab1:

    st.markdown(
        '<div class="phead">Retrieval Performance</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Accuracy Graph
    # --------------------------------------------------------

    accuracy_graph = os.path.join(
        EVAL_DIR,
        "graphs",
        "accuracy_comparison_2632.png"
    )

    if os.path.exists(accuracy_graph):

        st.image(
            accuracy_graph,
            caption="Accuracy Comparison — 2,632 Queries",
            use_container_width=True
        )

    else:

        st.warning(
            "Accuracy comparison graph not found."
        )


    # --------------------------------------------------------
    # Latency Graph
    # --------------------------------------------------------

    latency_graph = os.path.join(
        EVAL_DIR,
        "graphs",
        "latency_comparison_2632.png"
    )

    if os.path.exists(latency_graph):

        st.image(
            latency_graph,
            caption="Average Latency Comparison",
            use_container_width=True
        )

    else:

        st.warning(
            "Latency comparison graph not found."
        )


    # --------------------------------------------------------
    # Numerical Results
    # --------------------------------------------------------

    st.markdown(
        '<div class="phead">Numerical Results</div>',
        unsafe_allow_html=True
    )


    if not results_df.empty:

        display_df = results_df.copy()

        display_df["Accuracy (%)"] = (
            display_df["Accuracy (%)"]
            .round(2)
        )

        if "Average Latency (s)" in display_df.columns:

            display_df["Average Latency (s)"] = (
                display_df["Average Latency (s)"]
                .round(4)
            )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No evaluation result files were found."
        )


    # --------------------------------------------------------
    # Detailed Accuracy Summary
    # --------------------------------------------------------

    if not results_df.empty:

        st.markdown(
            '<div class="phead">Correct Predictions</div>',
            unsafe_allow_html=True
        )

        correct_df = results_df[
            [
                "Method",
                "Queries",
                "Correct",
                "Accuracy (%)"
            ]
        ].copy()

        correct_df["Accuracy (%)"] = (
            correct_df["Accuracy (%)"].round(2)
        )

        st.dataframe(
            correct_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2
# RETRIEVAL ABLATION
# ============================================================

with tab2:

    st.markdown(
        '<div class="phead">Retrieval Ablation Study</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="tracebox">

        <div class="ttitle">
            Retrieval Configurations
        </div>

        <div class="trow">
            <span class="tkey">BM25-only</span>
            <span class="tval">
                Sparse lexical retrieval
            </span>
        </div>

        <div class="trow">
            <span class="tkey">FAISS-only</span>
            <span class="tval">
                Dense semantic retrieval
            </span>
        </div>

        <div class="trow">
            <span class="tkey">Fair RRF</span>
            <span class="tval">
                FAISS + BM25 → RRF
            </span>
        </div>

        <div class="trow">
            <span class="tkey">Weighted RRF</span>
            <span class="tval">
                Weighted FAISS + BM25 → RRF
            </span>
        </div>

        <div class="trow">
            <span class="tkey">RRF + Cross-Encoder</span>
            <span class="tval">
                RRF → Cross-Encoder reranking
            </span>
        </div>

        <div class="trow">
            <span class="tkey">Top-10 Hybrid</span>
            <span class="tval">
                Top-10 hybrid retrieval configuration
            </span>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Retrieval Ablation Graph
    # --------------------------------------------------------

    retrieval_graph = os.path.join(
        EVAL_DIR,
        "graphs",
        "retrieval_ablation_comparison_2632.png"
    )

    if os.path.exists(retrieval_graph):

        st.image(
            retrieval_graph,
            caption="Retrieval Ablation — 2,632 Queries",
            use_container_width=True
        )

    else:

        st.warning(
            "Retrieval ablation graph not found."
        )


    # --------------------------------------------------------
    # Retrieval Ablation CSV
    # --------------------------------------------------------

    retrieval_csv = os.path.join(
        EVAL_DIR,
        "graphs",
        "retrieval_ablation_comparison_2632.csv"
    )

    if os.path.exists(retrieval_csv):

        st.markdown(
            '<div class="phead">Ablation Results</div>',
            unsafe_allow_html=True
        )

        retrieval_df = pd.read_csv(
            retrieval_csv
        )

        st.dataframe(
            retrieval_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Individual Method Summary
    # --------------------------------------------------------

    if not results_df.empty:

        st.markdown(
            '<div class="phead">Method-wise Comparison</div>',
            unsafe_allow_html=True
        )

        method_compare = results_df[
            [
                "Method",
                "Correct",
                "Queries",
                "Accuracy (%)"
            ]
        ].copy()

        method_compare["Accuracy (%)"] = (
            method_compare["Accuracy (%)"].round(2)
        )

        st.dataframe(
            method_compare,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 3
# COMPONENT ABLATION
# ============================================================

with tab3:

    st.markdown(
        '<div class="phead">Multi-Agent Component Ablation</div>',
        unsafe_allow_html=True
    )


    component_dir = os.path.join(
        EVAL_DIR,
        "ablation_study"
    )


    # --------------------------------------------------------
    # Ablation Graph
    # --------------------------------------------------------

    component_graph = os.path.join(
        component_dir,
        "ablation_comparison_100.png"
    )

    if os.path.exists(component_graph):

        st.image(
            component_graph,
            caption="Multi-Agent Component Ablation",
            use_container_width=True
        )

    else:

        st.warning(
            "Component ablation graph not found."
        )


    # --------------------------------------------------------
    # Ablation Summary
    # --------------------------------------------------------

    component_summary = os.path.join(
        component_dir,
        "ablation_study_summary_2632.csv"
    )

    if os.path.exists(component_summary):

        st.markdown(
            '<div class="phead">Component Ablation Results</div>',
            unsafe_allow_html=True
        )

        component_df = pd.read_csv(
            component_summary
        )

        st.dataframe(
            component_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Component ablation summary CSV not found."
        )


    # --------------------------------------------------------
    # Detailed Ablation Results
    # --------------------------------------------------------

    component_results = os.path.join(
        component_dir,
        "ablation_study_results_2632.csv"
    )

    if os.path.exists(component_results):

        with st.expander(
            "View Detailed Ablation Results"
        ):

            detailed_ablation_df = pd.read_csv(
                component_results
            )

            st.dataframe(
                detailed_ablation_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# TAB 4
# CONFUSION MATRICES
# ============================================================

with tab4:

    st.markdown(
        '<div class="phead">Confusion Matrices</div>',
        unsafe_allow_html=True
    )


    confusion_dir = os.path.join(
        EVAL_DIR,
        "confusion_matrices"
    )


    confusion_files = {
        "BM25-only":
            "confusion_matrix_bm25.png",

        "FAISS-only":
            "confusion_matrix_faiss.png",

        "Fair RRF":
            "confusion_matrix_fair_rrf.png",

        "Weighted RRF":
            "confusion_matrix_weighted_rrf.png",

        "RRF + Cross-Encoder":
            "confusion_matrix_hybrid.png",

        "Top-10 Hybrid":
            "confusion_matrix_top10_hybrid.png",
    }


    selected_method = st.selectbox(
        "Select Retrieval Configuration",
        list(confusion_files.keys())
    )


    selected_confusion = os.path.join(
        confusion_dir,
        confusion_files[selected_method]
    )


    if os.path.exists(selected_confusion):

        st.image(
            selected_confusion,
            caption=f"Confusion Matrix — {selected_method}",
            use_container_width=True
        )

    else:

        st.warning(
            f"Confusion matrix not found for {selected_method}."
        )


    # --------------------------------------------------------
    # CSV Confusion Matrix
    # --------------------------------------------------------

    confusion_csv_name = (
        confusion_files[selected_method]
        .replace(".png", ".csv")
    )

    confusion_csv = os.path.join(
        confusion_dir,
        confusion_csv_name
    )


    if os.path.exists(confusion_csv):

        st.markdown(
            '<div class="phead">Confusion Matrix Values</div>',
            unsafe_allow_html=True
        )

        confusion_df = pd.read_csv(
            confusion_csv,
            index_col=0
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )


# ============================================================
# TAB 5
# CLASSIFICATION REPORTS
# ============================================================

with tab5:

    st.markdown(
        '<div class="phead">Classification Reports</div>',
        unsafe_allow_html=True
    )


    reports_dir = os.path.join(
        EVAL_DIR,
        "classification_reports"
    )


    report_files = {
        "BM25-only":
            "classification_report_bm25.txt",

        "FAISS-only":
            "classification_report_faiss.txt",

        "Fair RRF":
            "classification_report_fair_rrf.txt",

        "Weighted RRF":
            "classification_report_weighted_rrf.txt",

        "RRF + Cross-Encoder":
            "classification_report_hybrid.txt",

        "Top-10 Hybrid":
            "classification_report_top10_hybrid.txt",
    }


    selected_report_method = st.selectbox(
        "Select Configuration",
        list(report_files.keys()),
        key="classification_report_method"
    )


    selected_report = os.path.join(
        reports_dir,
        report_files[selected_report_method]
    )


    if os.path.exists(selected_report):

        with open(
            selected_report,
            "r",
            encoding="utf-8"
        ) as f:

            report_text = f.read()


        st.code(
            report_text,
            language="text"
        )

    else:

        st.warning(
            f"Classification report not found for "
            f"{selected_report_method}."
        )


# ============================================================
# TAB 6
# STATISTICAL ANALYSIS
# ============================================================

with tab6:

    st.markdown(
        '<div class="phead">Statistical Significance Analysis</div>',
        unsafe_allow_html=True
    )


    statistical_dir = os.path.join(
        EVAL_DIR,
        "statistical_tests"
    )


    # --------------------------------------------------------
    # Main Statistical Test
    # --------------------------------------------------------

    statistical_csv = os.path.join(
        statistical_dir,
        "statistical_significance_tests.csv"
    )


    if os.path.exists(statistical_csv):

        st.markdown(
            '<div class="phead">McNemar Statistical Test</div>',
            unsafe_allow_html=True
        )

        statistical_df = pd.read_csv(
            statistical_csv
        )

        st.dataframe(
            statistical_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "McNemar's test evaluates paired prediction "
            "differences between retrieval configurations."
        )

    else:

        st.warning(
            "Statistical significance CSV not found."
        )


    # --------------------------------------------------------
    # V2 Statistical Test
    # --------------------------------------------------------

    statistical_csv_v2 = os.path.join(
        statistical_dir,
        "statistical_significance_tests_v2.csv"
    )


    if os.path.exists(statistical_csv_v2):

        st.markdown(
            '<div class="phead">Statistical Analysis — Version 2</div>',
            unsafe_allow_html=True
        )

        statistical_v2_df = pd.read_csv(
            statistical_csv_v2
        )

        st.dataframe(
            statistical_v2_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # TXT Results
    # --------------------------------------------------------

    statistical_txt = os.path.join(
        statistical_dir,
        "statistical_significance_tests.txt"
    )


    if os.path.exists(statistical_txt):

        with st.expander(
            "View Detailed Statistical Test Report"
        ):

            with open(
                statistical_txt,
                "r",
                encoding="utf-8"
            ) as f:

                statistical_text = f.read()


            st.code(
                statistical_text,
                language="text"
            )


# ============================================================
# ERROR ANALYSIS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="phead">🔎 Error Analysis</div>',
    unsafe_allow_html=True
)


error_dir = os.path.join(
    EVAL_DIR,
    "error_analysis"
)


error_files = {
    "BM25-only":
        "bm25_wrong_queries.csv",

    "FAISS-only":
        "faiss_wrong_queries.csv",

    "Fair RRF":
        "fair_rrf_wrong_queries.csv",

    "Weighted RRF":
        "weighted_rrf_wrong_queries.csv",

    "RRF + Cross-Encoder":
        "hybrid_wrong_queries.csv",

    "Top-10 Hybrid":
        "top10_hybrid_wrong_queries.csv",
}


error_method = st.selectbox(
    "Select Configuration for Error Analysis",
    list(error_files.keys()),
    key="error_analysis_method"
)


error_path = os.path.join(
    error_dir,
    error_files[error_method]
)


if os.path.exists(error_path):

    error_df = pd.read_csv(
        error_path
    )


    st.write(
        f"Incorrect predictions: "
        f"**{len(error_df):,}**"
    )


    st.dataframe(
        error_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        f"Error analysis file not found for {error_method}."
    )


# ============================================================
# LATENCY SUMMARY
# ============================================================

st.markdown(
    '<div class="phead">⏱️ Latency Analysis</div>',
    unsafe_allow_html=True
)


latency_summary_path = os.path.join(
    EVAL_DIR,
    "latency_analysis",
    "latency_summary.csv"
)


if os.path.exists(latency_summary_path):

    latency_summary_df = pd.read_csv(
        latency_summary_path
    )

    st.dataframe(
        latency_summary_df,
        use_container_width=True,
        hide_index=True
    )

else:

    # Build latency summary directly from evaluation files
    latency_rows = []

    for method, df in evaluation_data.items():

        latency = calculate_latency(df)

        latency_rows.append(
            {
                "Method": method,
                "Average Latency (s)": latency
            }
        )


    latency_df = pd.DataFrame(
        latency_rows
    )


    if not latency_df.empty:

        latency_df[
            "Average Latency (s)"
        ] = latency_df[
            "Average Latency (s)"
        ].round(4)


        st.dataframe(
            latency_df,
            use_container_width=True,
            hide_index=True
        )