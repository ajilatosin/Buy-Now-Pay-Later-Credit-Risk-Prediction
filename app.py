import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BNPL Credit Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d1117 100%);
    color: #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #1e3a5f;
  }
  section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
  }
  section[data-testid="stSidebar"] .stSlider > div > div {
    background: #1e3a5f !important;
  }

  /* Title */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 300;
    margin-top: 4px;
    margin-bottom: 1.5rem;
  }

  /* Decision card */
  .decision-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 12px 0;
    font-family: 'Syne', sans-serif;
  }
  .card-approve {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 1px solid #22c55e55;
    box-shadow: 0 0 40px #22c55e22;
  }
  .card-reject {
    background: linear-gradient(135deg, #3b0000 0%, #7f1d1d 100%);
    border: 1px solid #ef444455;
    box-shadow: 0 0 40px #ef444422;
  }
  .card-review {
    background: linear-gradient(135deg, #2d1a00 0%, #78350f 100%);
    border: 1px solid #f59e0b55;
    box-shadow: 0 0 40px #f59e0b22;
  }
  .decision-label {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 0;
  }
  .decision-sub {
    font-size: 1rem;
    font-weight: 400;
    opacity: 0.75;
    margin-top: 4px;
    font-family: 'DM Sans', sans-serif;
  }

  /* Metric cards */
  .metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-box .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
  }
  .metric-box .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 4px 0 0;
  }

  /* Risk factor pill */
  .factor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: #1e293b;
    border-radius: 8px;
    margin-bottom: 8px;
    border-left: 4px solid;
    font-size: 0.9rem;
  }
  .factor-good { border-color: #22c55e; }
  .factor-warn { border-color: #f59e0b; }
  .factor-bad  { border-color: #ef4444; }

  /* Section headers */
  .section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
  }

  /* Streamlit overrides */
  div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
  }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
  }
  .stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    color: #0f172a;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 14px;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  div[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load('bnpl_hybrid_model.pkl')
    except FileNotFoundError:
        return None

@st.cache_resource
def load_features():
    try:
        with open('feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return [
            'credit_score', 'debt_to_income_ratio', 'monthly_income',
            'purchase_amount', 'bnpl_installments', 'repayment_delay_days',
            'missed_payments', 'risk_score', 'age',
            'payment_compliance', 'debt_burden', 'risk_adjusted_credit',
            'high_risk_customer', 'severe_delinquent'
        ]

@st.cache_resource
def load_config():
    try:
        with open('threshold_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'approve_threshold': 0.3, 'reject_threshold': 0.7}

model        = load_model()
feature_names = load_features()
config        = load_config()

# ─── Helper Functions ────────────────────────────────────────────────────────────
def hybrid_decision(prob):
    if prob < config['approve_threshold']:
        return 'APPROVE', 'Auto-Approved'
    elif prob < config['reject_threshold']:
        return 'REVIEW', 'Manual Review Required'
    else:
        return 'REJECT', 'Auto-Rejected'


def create_features(age, credit_score, dti, income, purchase, installments, delay, missed, risk):
    monthly_payment   = purchase / max(installments, 1)
    payment_compliance = 1 - (missed / (installments + 1))
    debt_burden        = monthly_payment / (income + 1)
    risk_adjusted_credit = credit_score * (1 - risk / 100)
    high_risk          = 1 if (risk > 70 or credit_score < 580) else 0
    severe_delinquent  = 1 if delay > 30 else 0

    features = {
        'credit_score':         credit_score,
        'debt_to_income_ratio': dti,
        'monthly_income':       income,
        'purchase_amount':      purchase,
        'bnpl_installments':    installments,
        'repayment_delay_days': delay,
        'missed_payments':      missed,
        'risk_score':           risk,
        'age':                  age,
        'payment_compliance':   payment_compliance,
        'debt_burden':          debt_burden,
        'risk_adjusted_credit': risk_adjusted_credit,
        'high_risk_customer':   high_risk,
        'severe_delinquent':    severe_delinquent,
    }
    return pd.DataFrame([features])[feature_names]


def credit_score_label(score):
    if score >= 800: return "Exceptional", "#22c55e"
    if score >= 740: return "Very Good",   "#4ade80"
    if score >= 670: return "Good",        "#a3e635"
    if score >= 580: return "Fair",        "#f59e0b"
    return "Poor", "#ef4444"


def risk_color(prob):
    if prob < 0.3: return "#22c55e"
    if prob < 0.7: return "#f59e0b"
    return "#ef4444"


def gauge_chart(prob):
    color = risk_color(prob)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        number= {"suffix": "%", "font": {"size": 40, "color": color, "family": "Syne"}},
        gauge = {
            "axis":      {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar":       {"color": color, "thickness": 0.28},
            "bgcolor":   "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#052e16"},
                {"range": [30, 70], "color": "#2d1a00"},
                {"range": [70, 100],"color": "#3b0000"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
        title={"text": "Default Probability", "font": {"color": "#94a3b8", "size": 13, "family": "DM Sans"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=240,
        font_color="#e2e8f0",
    )
    return fig


def radar_chart(age, credit_score, dti, income, purchase, installments, delay, missed, risk):
    """Normalised radar of key risk dimensions (higher = riskier)."""
    categories = ["Credit Risk", "DTI Burden", "Delay Risk", "Missed Pmts", "Purchase Size"]
    vals = [
        max(0, (850 - credit_score) / 550),   # lower score → higher risk
        min(dti / 1.0, 1.0),
        min(delay / 90, 1.0),
        min(missed / max(installments, 1), 1.0),
        min(purchase / 10000, 1.0),
    ]
    vals_loop = vals + [vals[0]]
    cats_loop = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r     = vals_loop,
        theta = cats_loop,
        fill  = 'toself',
        fillcolor = "rgba(56,189,248,0.15)",
        line  = dict(color="#38bdf8", width=2),
        name  = "Risk Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30,41,59,0.8)",
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#475569", size=9), gridcolor="#334155"),
            angularaxis=dict(tickfont=dict(color="#94a3b8", size=11), gridcolor="#334155"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=30, b=30, l=40, r=40),
        height=280,
    )
    return fig


def feature_bar_chart(features_df):
    vals  = features_df.iloc[0].to_dict()
    items = {
        "Credit Score":     vals.get("credit_score", 0) / 850,
        "DTI Ratio":        1 - vals.get("debt_to_income_ratio", 0),
        "Pay Compliance":   vals.get("payment_compliance", 0),
        "Risk-Adj Credit":  vals.get("risk_adjusted_credit", 0) / 850,
        "Debt Burden":      1 - min(vals.get("debt_burden", 0) * 10, 1),
    }
    labels = list(items.keys())
    values = list(items.values())
    colors = ["#22c55e" if v > 0.6 else "#f59e0b" if v > 0.35 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
        textfont=dict(color="#cbd5e1", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        xaxis=dict(range=[0,1.15], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", size=12)),
        margin=dict(t=10, b=10, l=10, r=60),
        height=220,
        bargap=0.3,
    )
    return fig


def risk_factor_breakdown(credit_score, dti, delay, missed, installments, risk, income, purchase):
    factors = []
    cs_label, _ = credit_score_label(credit_score)
    factors.append({
        "name":  f"Credit Score — {cs_label} ({credit_score})",
        "level": "good" if credit_score >= 670 else ("warn" if credit_score >= 580 else "bad"),
        "note":  "Strong credit history" if credit_score >= 670 else ("Moderate risk" if credit_score >= 580 else "High default risk"),
    })
    factors.append({
        "name":  f"Debt-to-Income — {dti:.0%}",
        "level": "good" if dti < 0.35 else ("warn" if dti < 0.5 else "bad"),
        "note":  "Healthy leverage" if dti < 0.35 else ("Moderate burden" if dti < 0.5 else "Over-leveraged"),
    })
    factors.append({
        "name":  f"Repayment Delay — {delay}d",
        "level": "good" if delay == 0 else ("warn" if delay <= 30 else "bad"),
        "note":  "No delays" if delay == 0 else ("Minor delay" if delay <= 30 else "Severe delinquency"),
    })
    m_rate = missed / max(installments, 1)
    factors.append({
        "name":  f"Missed Payments — {missed}/{installments}",
        "level": "good" if m_rate == 0 else ("warn" if m_rate < 0.2 else "bad"),
        "note":  "Clean payment record" if m_rate == 0 else ("Occasional misses" if m_rate < 0.2 else "Poor payment history"),
    })
    aff = (purchase / installments) / max(income, 1)
    factors.append({
        "name":  f"Affordability — {aff:.1%} of income/month",
        "level": "good" if aff < 0.1 else ("warn" if aff < 0.25 else "bad"),
        "note":  "Easily affordable" if aff < 0.1 else ("Manageable" if aff < 0.25 else "Stretched finances"),
    })
    return factors


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Customer Profile")
    st.markdown("---")

    st.markdown("**👤 Demographics**")
    age = st.number_input("Age", 18, 100, 30, help="Customer age in years")

    st.markdown("**💰 Financial Health**")
    credit_score   = st.number_input("Credit Score", 300, 850, 650)
    cs_lbl, cs_col = credit_score_label(credit_score)
    st.markdown(f"<span style='color:{cs_col};font-size:0.82rem;font-weight:600;'>● {cs_lbl}</span>", unsafe_allow_html=True)

    monthly_income = st.number_input("Monthly Income ($)", 0, 200_000, 5000, step=500)
    dti            = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01,
                               help="Total monthly debt / gross monthly income")
    risk_score     = st.slider("Risk Score", 0, 400, 200,
                               help="Internal risk score from 0 (lowest) to 400 (highest)")

    st.markdown("**🛒 Purchase Details**")
    purchase_amount   = st.number_input("Purchase Amount ($)", 0, 50_000, 500, step=50)
    bnpl_installments = st.selectbox("Installments", [2, 3, 4, 6, 8, 10, 12])

    monthly_pmt = purchase_amount / max(bnpl_installments, 1)
    st.markdown(f"<span style='color:#94a3b8;font-size:0.82rem;'>📅 ~${monthly_pmt:,.0f}/month</span>", unsafe_allow_html=True)

    st.markdown("**📋 Repayment History**")
    missed_payments = st.number_input("Missed Payments", 0, bnpl_installments, 0)
    repayment_delay = st.number_input("Repayment Delay (days)", 0, 365, 0)

    st.markdown("---")
    assess_clicked = st.button("⚡ Assess Credit Risk", type="primary")
    st.markdown("<p style='text-align:center;color:#475569;font-size:0.75rem;margin-top:8px;'>Powered by Hybrid ML Model</p>", unsafe_allow_html=True)


# ─── Main Panel ──────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">💳 BNPL Credit Risk Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real-time creditworthiness assessment for Buy Now Pay Later decisions</p>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ **Model files not found.** Place `bnpl_hybrid_model.pkl`, `feature_names.pkl`, and `threshold_config.json` in the same directory as this app.")
    st.info("The UI is fully functional — connect your model files to enable live predictions.")
    st.stop()

# ─── Live Preview Cards (always visible) ─────────────────────────────────────────
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    cs_lbl2, cs_col2 = credit_score_label(credit_score)
    st.metric("Credit Score", credit_score, cs_lbl2)
with col_b:
    st.metric("Monthly Income", f"${monthly_income:,}")
with col_c:
    st.metric("Purchase / Month", f"${monthly_pmt:,.0f}")
with col_d:
    st.metric("DTI Ratio", f"{dti:.0%}")

st.markdown("---")

# ─── Assessment Results ───────────────────────────────────────────────────────────
if assess_clicked:
    with st.spinner("Running risk model…"):
        features = create_features(
            age, credit_score, dti, monthly_income,
            purchase_amount, bnpl_installments,
            repayment_delay, missed_payments, risk_score
        )
        prob            = model.predict_proba(features)[0, 1]
        decision, action = hybrid_decision(prob)

    # ── Decision banner ──
    card_class = {"APPROVE": "card-approve", "REJECT": "card-reject", "REVIEW": "card-review"}[decision]
    icons      = {"APPROVE": "✅", "REJECT": "❌", "REVIEW": "⚠️"}
    colors_dec = {"APPROVE": "#22c55e", "REJECT": "#ef4444", "REVIEW": "#f59e0b"}
    st.markdown(f"""
    <div class="decision-card {card_class}">
      <p class="decision-label" style="color:{colors_dec[decision]};">{icons[decision]} {decision}</p>
      <p class="decision-sub">{action}</p>
      <p style="font-family:'DM Sans';font-size:0.8rem;color:#64748b;margin-top:8px;">
        Assessed {datetime.now().strftime('%d %b %Y, %H:%M')}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ──
    left, mid, right = st.columns([1.1, 1, 1])

    with left:
        st.markdown('<p class="section-head">Default Probability</p>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
        approve_t = config['approve_threshold']
        reject_t  = config['reject_threshold']
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b;margin-top:-10px;'>
          <span>✅ &lt;{approve_t:.0%} Auto-Approve</span>
          <span>⚠️ {approve_t:.0%}–{reject_t:.0%} Review</span>
          <span>❌ &gt;{reject_t:.0%} Reject</span>
        </div>
        """, unsafe_allow_html=True)

    with mid:
        st.markdown('<p class="section-head">Risk Radar</p>', unsafe_allow_html=True)
        st.plotly_chart(
            radar_chart(age, credit_score, dti, monthly_income, purchase_amount,
                        bnpl_installments, repayment_delay, missed_payments, risk_score),
            use_container_width=True, config={"displayModeBar": False}
        )

    with right:
        st.markdown('<p class="section-head">Positive Signal Strength</p>', unsafe_allow_html=True)
        st.plotly_chart(feature_bar_chart(features), use_container_width=True, config={"displayModeBar": False})

    # ── Risk Factor Breakdown ──
    st.markdown('<p class="section-head">Risk Factor Breakdown</p>', unsafe_allow_html=True)
    factors = risk_factor_breakdown(
        credit_score, dti, repayment_delay, missed_payments,
        bnpl_installments, risk_score, monthly_income, purchase_amount
    )
    cols_f = st.columns(len(factors))
    for col, f in zip(cols_f, factors):
        border = {"good": "#22c55e", "warn": "#f59e0b", "bad": "#ef4444"}[f["level"]]
        icon   = {"good": "✅",       "warn": "⚠️",      "bad": "❌"}[f["level"]]
        with col:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:10px;border-top:3px solid {border};
                        padding:14px;text-align:center;height:110px;'>
              <div style='font-size:1.4rem;'>{icon}</div>
              <div style='font-size:0.78rem;color:#94a3b8;margin:4px 0 2px;font-weight:600;'>{f["name"].split("—")[0].strip()}</div>
              <div style='font-size:0.82rem;color:{border};'>{f["note"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Key Metrics Row ──
    st.markdown('<p class="section-head">Derived Metrics</p>', unsafe_allow_html=True)
    monthly_payment    = purchase_amount / max(bnpl_installments, 1)
    payment_compliance = 1 - (missed_payments / (bnpl_installments + 1))
    debt_burden        = monthly_payment / (monthly_income + 1)
    risk_adj_credit    = credit_score * (1 - risk_score / 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Payment Compliance", f"{payment_compliance:.1%}", delta=f"{'Good' if payment_compliance > 0.8 else 'Poor'}")
    m2.metric("Debt Burden", f"{debt_burden:.2%}", delta=f"{'Healthy' if debt_burden < 0.1 else 'High'}", delta_color="inverse")
    m3.metric("Risk-Adj Credit", f"{risk_adj_credit:.0f}", delta=f"of 850 max")
    m4.metric("Effective Rate", f"{missed_payments/max(bnpl_installments,1):.0%}", delta="Miss rate", delta_color="inverse")

    # ── Model Explainability ──
    with st.expander("📊 Model Inputs & Feature Values", expanded=False):
        st.dataframe(
            features.T.rename(columns={0: "Value"}).style
              .background_gradient(cmap="RdYlGn", axis=0)
              .format("{:.4f}"),
            use_container_width=True,
        )

    # ── Recommendation Box ──
    st.markdown('<p class="section-head">Analyst Recommendation</p>', unsafe_allow_html=True)
    recs = {
        "APPROVE": [
            f"✅ Customer qualifies for up to **${purchase_amount:,}** across **{bnpl_installments} installments**.",
            f"📅 Monthly obligation of **${monthly_payment:,.0f}** is within acceptable range.",
            "🔄 Schedule standard 30-day follow-up after first payment.",
        ],
        "REVIEW": [
            "🔍 Escalate to manual underwriting team for human review.",
            f"📞 Consider requesting additional income documentation (income: ${monthly_income:,}/mo).",
            f"⚖️ Probability {prob:.1%} sits in the **grey zone** — additional context may shift decision.",
        ],
        "REJECT": [
            f"❌ Default probability of **{prob:.1%}** exceeds rejection threshold of **{config['reject_threshold']:.0%}**.",
            f"📉 Credit score of **{credit_score}** and/or DTI of **{dti:.0%}** are primary risk drivers.",
            "💡 Suggest customer returns after 6 months of on-time payments and credit improvement.",
        ],
    }
    rec_color = colors_dec[decision]
    rec_lines = recs[decision]
    rec_html  = "".join(f"<p style='margin:6px 0;font-size:0.9rem;color:#cbd5e1;'>{line}</p>" for line in rec_lines)
    st.markdown(f"""
    <div style='background:#1e293b;border-left:4px solid {rec_color};
                border-radius:0 12px 12px 0;padding:20px 24px;'>
      {rec_html}
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div style='text-align:center;padding:80px 40px;'>
      <div style='font-size:4rem;margin-bottom:16px;'>🏦</div>
      <p style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;color:#475569;'>
        Fill in the customer profile and click <br><span style='color:#38bdf8;'>⚡ Assess Credit Risk</span>
      </p>
      <p style='color:#334155;font-size:0.9rem;margin-top:8px;'>
        The engine will return a live decision, risk radar, gauge, and analyst recommendation.
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.78rem;'>"
    "BNPL Credit Risk Engine · Hybrid ML Model · "
    f"Thresholds: Approve &lt;{config['approve_threshold']:.0%} · "
    f"Reject &gt;{config['reject_threshold']:.0%}"
    "</p>",
    unsafe_allow_html=True
)
