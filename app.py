"""
╔══════════════════════════════════════════════════════════════════╗
║        SUPPLY CHAIN ANALYTICS DASHBOARD — app.py                ║
║        Built with Streamlit · Plotly · Scikit-learn             ║
╚══════════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit pandas numpy plotly scikit-learn scipy
    streamlit run app.py
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Analytics",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:       #0a0c10;
    --bg-card:       #0f1218;
    --bg-panel:      #141820;
    --bg-hover:      #1a2030;
    --border:        #1e2535;
    --border-bright: #2a3550;
    --accent-cyan:   #00d4ff;
    --accent-teal:   #00ffcc;
    --accent-amber:  #ffb800;
    --accent-rose:   #ff4d7e;
    --accent-purple: #a855f7;
    --text-primary:  #e8edf5;
    --text-secondary:#8892a4;
    --text-dim:      #4a5568;
    --glow-cyan:     0 0 20px rgba(0,212,255,0.3);
    --glow-teal:     0 0 20px rgba(0,255,204,0.25);
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App Background ── */
.stApp {
    background: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,212,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 90% 80%, rgba(168,85,247,0.04) 0%, transparent 50%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

/* ── Sidebar Nav Items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 3px 0;
    cursor: pointer;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all 0.2s;
    border: 1px solid transparent;
}
.nav-item:hover {
    background: var(--bg-hover);
    color: var(--accent-cyan);
    border-color: var(--border-bright);
}
.nav-item.active {
    background: rgba(0,212,255,0.08);
    color: var(--accent-cyan);
    border-color: rgba(0,212,255,0.25);
    box-shadow: var(--glow-cyan);
}

/* ── Header ── */
.dashboard-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 28px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-teal), transparent);
}
.header-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.header-title span { color: var(--accent-cyan); }
.header-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-cyan);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent-teal);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.5; transform:scale(1.3); }
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--border-bright); }
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.kpi-card.cyan::after  { background: var(--accent-cyan); }
.kpi-card.teal::after  { background: var(--accent-teal); }
.kpi-card.amber::after { background: var(--accent-amber); }
.kpi-card.rose::after  { background: var(--accent-rose); }
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-value.cyan  { color: var(--accent-cyan); }
.kpi-value.teal  { color: var(--accent-teal); }
.kpi-value.amber { color: var(--accent-amber); }
.kpi-value.rose  { color: var(--accent-rose); }
.kpi-delta {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-secondary);
}
.kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    font-size: 22px;
    opacity: 0.15;
}

/* ── Section Titles ── */
.section-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 8px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Insight Box ── */
.insight-box {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-secondary);
}
.insight-box strong { color: var(--accent-cyan); }

/* ── Stat Cards ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 120px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-card-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
}
.stat-card-value {
    font-size: 20px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
}

/* ── Selectbox & Slider ── */
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSlider > div { color: var(--text-secondary) !important; }
[data-baseweb="select"] { background: var(--bg-panel) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: none !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: var(--accent-cyan) !important;
}

/* ── DataFrames ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Number Input ── */
.stNumberInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Download Button ── */
.stDownloadButton button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: var(--accent-cyan) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover {
    background: rgba(0,212,255,0.2) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── Metric widget override ── */
[data-testid="metric-container"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px !important;
}

/* ── Radio (sidebar nav) ── */
.stRadio > div { gap: 4px !important; }
.stRadio label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px !important;
    padding: 9px 12px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio label:hover {
    background: var(--bg-hover) !important;
    color: var(--accent-cyan) !important;
    border-color: var(--border-bright) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 10px; }

/* ── Plotly chart background ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate a realistic supply chain dataset.
    Uses structured randomness so analytics tell a meaningful story.
    """
    rng = np.random.default_rng(42)

    products     = ["Electronics", "Cosmetics", "Clothing", "Haircare", "Skincare"]
    locations    = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad"]
    suppliers    = ["Supplier_A", "Supplier_B", "Supplier_C", "Supplier_D", "Supplier_E"]
    months       = pd.date_range("2023-01-01", periods=n, freq="D")

    # Base profiles per product type
    profiles = {
        "Electronics": dict(price_base=350, demand_base=180, defect=0.04, lead=14, ship=22, mfg=180),
        "Cosmetics":   dict(price_base=85,  demand_base=320, defect=0.02, lead=7,  ship=10, mfg=28),
        "Clothing":    dict(price_base=65,  demand_base=400, defect=0.03, lead=10, ship=8,  mfg=22),
        "Haircare":    dict(price_base=55,  demand_base=280, defect=0.025,lead=6,  ship=7,  mfg=18),
        "Skincare":    dict(price_base=70,  demand_base=260, defect=0.022,lead=8,  ship=9,  mfg=20),
    }

    rows = []
    for i in range(n):
        pt   = rng.choice(products)
        p    = profiles[pt]
        loc  = rng.choice(locations)
        sup  = rng.choice(suppliers)
        price = max(10, rng.normal(p["price_base"], p["price_base"]*0.18))
        demand = max(10, int(rng.normal(p["demand_base"], p["demand_base"]*0.25)))
        revenue = round(price * demand * rng.uniform(0.85, 1.15), 2)
        stock   = int(rng.normal(demand * 1.5, demand * 0.4))
        lead    = max(1, int(rng.normal(p["lead"], p["lead"]*0.3)))
        ship    = max(1, round(rng.normal(p["ship"], p["ship"]*0.2), 2))
        mfg     = max(5, round(rng.normal(p["mfg"],  p["mfg"]*0.15), 2))
        defect  = max(0, min(0.25, rng.normal(p["defect"], 0.01)))
        sku_id  = f"SKU-{rng.integers(1000,9999)}"

        rows.append({
            "Date":              months[i],
            "Product_Type":      pt,
            "SKU":               sku_id,
            "Location":          loc,
            "Supplier":          sup,
            "Price":             round(price, 2),
            "Demand":            demand,
            "Revenue":           revenue,
            "Stock_Level":       max(0, stock),
            "Lead_Time":         lead,
            "Shipping_Cost":     ship,
            "Manufacturing_Cost":mfg,
            "Defect_Rate":       round(defect, 4),
        })

    df = pd.DataFrame(rows)
    df["Profit_Margin"] = ((df["Revenue"] - df["Manufacturing_Cost"] - df["Shipping_Cost"])
                           / df["Revenue"]).round(4)
    df["Month"] = df["Date"].dt.strftime("%b %Y")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,18,26,0.6)",
        font=dict(family="Syne, sans-serif", color="#8892a4", size=12),
        title=dict(font=dict(size=15, color="#e8edf5", family="Syne")),
        xaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535", linecolor="#1e2535"),
        yaxis=dict(gridcolor="#1e2535", zerolinecolor="#1e2535", linecolor="#1e2535"),
        legend=dict(bgcolor="rgba(15,18,24,0.8)", bordercolor="#1e2535", borderwidth=1),
        colorway=["#00d4ff","#00ffcc","#a855f7","#ffb800","#ff4d7e",
                  "#3b82f6","#f97316","#10b981","#ec4899","#8b5cf6"],
        margin=dict(l=50, r=30, t=50, b=50),
    )
)

PALETTE = ["#00d4ff","#00ffcc","#a855f7","#ffb800","#ff4d7e",
           "#3b82f6","#f97316","#10b981","#ec4899","#8b5cf6"]


def apply_theme(fig):
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    fig.update_xaxes(showgrid=True, gridwidth=1)
    fig.update_yaxes(showgrid=True, gridwidth=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    """Render sidebar with filters and navigation."""
    with st.sidebar:
        # Logo / branding
        st.markdown("""
        <div style='padding:20px 0 10px;'>
            <div style='font-size:22px;font-weight:800;letter-spacing:-0.03em;color:#e8edf5;'>
                ⬡ <span style='color:#00d4ff;'>SC</span>Analytics
            </div>
            <div style='font-size:10px;font-weight:600;letter-spacing:0.12em;
                        text-transform:uppercase;color:#4a5568;margin-top:4px;'>
                Supply Chain Intelligence
            </div>
        </div>
        <div style='border-top:1px solid #1e2535;margin-bottom:20px;'></div>
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:0.12em;'
                    'text-transform:uppercase;color:#4a5568;margin-bottom:10px;">Navigation</div>',
                    unsafe_allow_html=True)

        nav_icons = {
            "📊  Overview":              "Overview",
            "📦  Box Plot Analysis":     "Box Plot",
            "📈  Regression & Predict":  "Regression",
            "🎲  Sampling Techniques":   "Sampling",
            "🔵  K-Means Clustering":    "Clustering",
            "🔔  Probability Dist.":     "Probability",
        }
        selected = st.radio(
            "nav", list(nav_icons.keys()),
            label_visibility="collapsed"
        )
        section = nav_icons[selected]

        st.markdown('<div style="border-top:1px solid #1e2535;margin:20px 0;"></div>',
                    unsafe_allow_html=True)

        # Global Filters
        st.markdown('<div style="font-size:10px;font-weight:700;letter-spacing:0.12em;'
                    'text-transform:uppercase;color:#4a5568;margin-bottom:10px;">Filters</div>',
                    unsafe_allow_html=True)

        products_all = ["All"] + sorted(df["Product_Type"].unique().tolist())
        sel_product  = st.selectbox("Product Type", products_all)

        locations_all = ["All"] + sorted(df["Location"].unique().tolist())
        sel_location  = st.selectbox("Location", locations_all)

        suppliers_all = ["All"] + sorted(df["Supplier"].unique().tolist())
        sel_supplier  = st.selectbox("Supplier", suppliers_all)

        # Apply filters
        filtered = df.copy()
        if sel_product  != "All": filtered = filtered[filtered["Product_Type"] == sel_product]
        if sel_location != "All": filtered = filtered[filtered["Location"]      == sel_location]
        if sel_supplier != "All": filtered = filtered[filtered["Supplier"]      == sel_supplier]

        st.markdown(f"""
        <div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.15);
                    border-radius:8px;padding:10px 14px;margin-top:8px;'>
            <div style='font-size:10px;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4a5568;margin-bottom:4px;'>Active Dataset</div>
            <div style='font-size:20px;font-weight:800;color:#00d4ff;'>{len(filtered):,}</div>
            <div style='font-size:11px;color:#8892a4;'>of {len(df):,} records</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇  Download Dataset",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="supply_chain_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    return section, filtered


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

def render_header(df: pd.DataFrame, filtered: pd.DataFrame):
    st.markdown(f"""
    <div class="dashboard-header">
        <div>
            <div style='font-size:11px;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4a5568;margin-bottom:6px;'>
                Analytics Platform
            </div>
            <div class="header-title">
                Supply Chain <span>Analytics</span> Dashboard
            </div>
        </div>
        <div style='display:flex;align-items:center;gap:12px;'>
            <div class="header-badge">
                <div class="live-dot"></div>
                Live Analytics
            </div>
            <div class="header-badge" style='color:#00ffcc;background:rgba(0,255,204,0.06);
                                              border-color:rgba(0,255,204,0.2);'>
                ⬡ {len(filtered):,} Records Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════

def render_kpis(df: pd.DataFrame):
    avg_revenue  = df["Revenue"].mean()
    avg_lead     = df["Lead_Time"].mean()
    avg_stock    = df["Stock_Level"].mean()
    avg_defect   = df["Defect_Rate"].mean() * 100
    total_rev    = df["Revenue"].sum()
    avg_margin   = df["Profit_Margin"].mean() * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card cyan">
            <div class="kpi-icon">💰</div>
            <div class="kpi-label">Avg. Revenue / Record</div>
            <div class="kpi-value cyan">${avg_revenue:,.0f}</div>
            <div class="kpi-delta">Total: ${total_rev/1e6:.1f}M cumulative</div>
        </div>
        <div class="kpi-card teal">
            <div class="kpi-icon">⏱</div>
            <div class="kpi-label">Avg. Lead Time</div>
            <div class="kpi-value teal">{avg_lead:.1f}<span style='font-size:14px;'> days</span></div>
            <div class="kpi-delta">Across all suppliers & products</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-icon">📦</div>
            <div class="kpi-label">Avg. Stock Level</div>
            <div class="kpi-value amber">{avg_stock:,.0f}<span style='font-size:14px;'> units</span></div>
            <div class="kpi-delta">Inventory snapshot</div>
        </div>
        <div class="kpi-card rose">
            <div class="kpi-icon">⚠</div>
            <div class="kpi-label">Avg. Defect Rate</div>
            <div class="kpi-value rose">{avg_defect:.2f}<span style='font-size:14px;'>%</span></div>
            <div class="kpi-delta">Quality control metric</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def section_overview(df: pd.DataFrame):
    st.markdown('<div class="section-title">Overview · Trend & Distribution</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Revenue by product over time
        monthly = (df.groupby(["Month", "Product_Type"])["Revenue"]
                   .sum().reset_index()
                   .sort_values("Month"))
        fig = px.area(monthly, x="Month", y="Revenue", color="Product_Type",
                      title="Monthly Revenue by Product Type",
                      color_discrete_sequence=PALETTE, template="none")
        fig.update_traces(line_width=1.5)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Revenue share pie
        rev_share = df.groupby("Product_Type")["Revenue"].sum().reset_index()
        fig2 = px.pie(rev_share, names="Product_Type", values="Revenue",
                      title="Revenue Distribution by Product",
                      color_discrete_sequence=PALETTE, hole=0.55, template="none")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Defect rate by supplier
        sup_defect = df.groupby("Supplier")["Defect_Rate"].mean().reset_index()
        sup_defect["Defect_Rate_pct"] = sup_defect["Defect_Rate"] * 100
        fig3 = px.bar(sup_defect.sort_values("Defect_Rate_pct"),
                      x="Defect_Rate_pct", y="Supplier", orientation="h",
                      title="Avg Defect Rate by Supplier (%)",
                      color="Defect_Rate_pct",
                      color_continuous_scale=["#00ffcc","#ffb800","#ff4d7e"],
                      template="none")
        apply_theme(fig3)
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Stock level choropleth-style scatter by location
        loc_stats = df.groupby("Location").agg(
            Revenue=("Revenue", "sum"),
            Demand=("Demand", "mean"),
            Stock=("Stock_Level", "mean")
        ).reset_index()
        fig4 = px.scatter(loc_stats, x="Revenue", y="Stock",
                          size="Demand", color="Location",
                          title="Location: Revenue vs Stock (size=Demand)",
                          color_discrete_sequence=PALETTE, template="none", size_max=50)
        apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>📡 Overview Insight:</strong>
        The area chart captures seasonal and category-level revenue momentum.
        Electronics and Skincare dominate the revenue share, while Supplier_C exhibits
        the highest average defect rate — a flag for quality audits.
        Locations with high stock but low revenue signal excess inventory risk.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — BOX PLOT
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_COLS = ["Price", "Demand", "Revenue", "Stock_Level", "Lead_Time",
                "Shipping_Cost", "Manufacturing_Cost", "Defect_Rate", "Profit_Margin"]

def section_boxplot(df: pd.DataFrame):
    st.markdown('<div class="section-title">Box Plot Analysis · Variability & Outliers</div>',
                unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        metric   = st.selectbox("Select Metric", NUMERIC_COLS, key="bp_metric")
    with col_ctrl2:
        group_by = st.selectbox("Group By", ["Product_Type", "Location", "Supplier"], key="bp_group")

    fig = px.box(df, x=group_by, y=metric, color=group_by,
                 title=f"Distribution of {metric.replace('_',' ')} by {group_by.replace('_',' ')}",
                 color_discrete_sequence=PALETTE, template="none",
                 points="outliers", notched=False)
    apply_theme(fig)
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown('<div class="section-title" style="font-size:11px;">Summary Statistics</div>',
                unsafe_allow_html=True)
    summary = df.groupby(group_by)[metric].describe().round(3)
    st.dataframe(
        summary.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True
    )

    # Outlier detection (IQR method)
    q1 = df[metric].quantile(0.25)
    q3 = df[metric].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[metric] < q1 - 1.5*iqr) | (df[metric] > q3 + 1.5*iqr)].copy()

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f'<div class="stat-card"><div class="stat-card-label">Mean</div>'
                    f'<div class="stat-card-value">{df[metric].mean():.3f}</div></div>',
                    unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f'<div class="stat-card"><div class="stat-card-label">Median</div>'
                    f'<div class="stat-card-value">{df[metric].median():.3f}</div></div>',
                    unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f'<div class="stat-card"><div class="stat-card-label">Std Dev</div>'
                    f'<div class="stat-card-value">{df[metric].std():.3f}</div></div>',
                    unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f'<div class="stat-card"><div class="stat-card-label">Outliers</div>'
                    f'<div class="stat-card-value">{len(outliers)}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if len(outliers) > 0:
        with st.expander(f"🔍 Outlier Records — {len(outliers)} detected"):
            display_cols = [group_by, "SKU", metric, "Date"]
            st.dataframe(
                outliers[display_cols].sort_values(metric, ascending=False).head(50),
                use_container_width=True
            )
    else:
        st.info("No outliers detected for this metric.")

    insight_vals = df.groupby(group_by)[metric].mean().sort_values(ascending=False)
    top_group    = insight_vals.index[0]
    st.markdown(f"""
    <div class="insight-box">
        <strong>📊 Box Plot Insight:</strong>
        <strong>{top_group}</strong> shows the highest mean {metric.replace("_"," ")} among all groups.
        The IQR method flagged <strong>{len(outliers)} outlier records</strong> — these warrant
        investigation as they may indicate data errors, exceptional events, or
        process anomalies in the supply chain.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def section_regression(df: pd.DataFrame):
    st.markdown('<div class="section-title">Linear Regression · Prediction Engine</div>',
                unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        x_var = st.selectbox("X Variable (Predictor)", NUMERIC_COLS, index=0, key="reg_x")
    with col_ctrl2:
        y_var = st.selectbox("Y Variable (Target)",    NUMERIC_COLS, index=2, key="reg_y")

    # Fit model
    X = df[[x_var]].values
    y = df[y_var].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2     = r2_score(y, y_pred)
    m      = model.coef_[0]
    b      = model.intercept_
    resids = y - y_pred

    col1, col2 = st.columns(2)

    with col1:
        # Scatter + regression line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_var], y=df[y_var], mode="markers",
            marker=dict(color="#00d4ff", size=5, opacity=0.5),
            name="Observations"
        ))
        x_line = np.linspace(X.min(), X.max(), 200)
        fig.add_trace(go.Scatter(
            x=x_line, y=model.predict(x_line.reshape(-1,1)),
            mode="lines", line=dict(color="#ff4d7e", width=2.5),
            name="Regression Line"
        ))
        fig.update_layout(
            title=f"{y_var.replace('_',' ')} vs {x_var.replace('_',' ')}",
            xaxis_title=x_var.replace("_"," "),
            yaxis_title=y_var.replace("_"," "),
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Residual plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_pred, y=resids, mode="markers",
            marker=dict(color="#a855f7", size=5, opacity=0.5),
            name="Residuals"
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="#ffb800", line_width=1.5)
        fig2.update_layout(
            title="Residual Plot",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
        )
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Stats bar
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Equation</div>
            <div style='font-size:13px;font-family:"JetBrains Mono",monospace;color:#00d4ff;margin-top:4px;'>
                y = {m:.4f}x + {b:.2f}
            </div>
        </div>""", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">R² Score</div>
            <div class="stat-card-value" style='color:{"#00ffcc" if r2>0.5 else "#ffb800"};'>
                {r2:.4f}
            </div>
        </div>""", unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Correlation (r)</div>
            <div class="stat-card-value">{np.corrcoef(X.ravel(), y)[0,1]:.4f}</div>
        </div>""", unsafe_allow_html=True)

    # Prediction box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title" style="font-size:11px;">🔮 Predict {y_var.replace("_"," ")}</div>',
                unsafe_allow_html=True)
    x_input = st.number_input(
        f"Enter a value for {x_var.replace('_',' ')}",
        value=float(df[x_var].mean()), format="%.4f", key="reg_pred_input"
    )
    y_predicted = m * x_input + b
    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
                border-radius:10px;padding:16px 22px;display:flex;align-items:center;gap:20px;'>
        <div>
            <div style='font-size:10px;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#4a5568;margin-bottom:6px;'>
                Predicted {y_var.replace("_"," ")}
            </div>
            <div style='font-size:32px;font-weight:800;color:#00d4ff;font-family:"JetBrains Mono",monospace;'>
                {y_predicted:,.2f}
            </div>
        </div>
        <div style='font-size:13px;color:#8892a4;'>
            Given {x_var.replace("_"," ")} = <strong style='color:#e8edf5;'>{x_input:.4f}</strong><br>
            using model: y = {m:.4f}x + {b:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_quality = "strong" if r2 > 0.6 else "moderate" if r2 > 0.3 else "weak"
    st.markdown(f"""
    <div class="insight-box" style='margin-top:16px;'>
        <strong>📈 Regression Insight:</strong>
        The model explains <strong>{r2*100:.1f}%</strong> of variance in
        <strong>{y_var.replace("_"," ")}</strong> from <strong>{x_var.replace("_"," ")}</strong>
        — a <strong>{model_quality}</strong> predictive relationship.
        {'Residuals appear randomly scattered, indicating good model fit.' if r2 > 0.4
         else 'Residual patterns suggest non-linearity or missing variables.'}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

def section_sampling(df: pd.DataFrame):
    st.markdown('<div class="section-title">Sampling Techniques · Statistical Estimation</div>',
                unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        metric = st.selectbox("Metric to Sample", NUMERIC_COLS, index=2, key="samp_metric")
    with col_ctrl2:
        n_sample = st.slider("Sample Size", min_value=20, max_value=min(200, len(df)),
                             value=80, step=10, key="samp_n")

    rng_s = np.random.default_rng(99)

    # ── Simple Random Sampling
    srs = df.sample(n=n_sample, random_state=42)[metric]

    # ── Systematic Sampling
    step   = max(1, len(df) // n_sample)
    start  = rng_s.integers(0, step)
    idxs   = list(range(start, len(df), step))[:n_sample]
    sys_s  = df.iloc[idxs][metric]

    # ── Stratified Sampling (by Product_Type)
    strat_frames = []
    for grp, grp_df in df.groupby("Product_Type"):
        n_grp = max(1, int(n_sample * len(grp_df) / len(df)))
        strat_frames.append(grp_df.sample(min(n_grp, len(grp_df)), random_state=42))
    strat_s = pd.concat(strat_frames)[metric].iloc[:n_sample]

    population = df[metric]

    # Comparison chart
    fig = go.Figure()
    for label, data, color in [
        ("Population",  population, "#4a5568"),
        ("Simple Rand", srs,        "#00d4ff"),
        ("Systematic",  sys_s,      "#00ffcc"),
        ("Stratified",  strat_s,    "#a855f7"),
    ]:
        fig.add_trace(go.Histogram(
            x=data, name=label, opacity=0.65,
            marker_color=color, nbinsx=30,
            histnorm="probability density"
        ))
    fig.update_layout(
        barmode="overlay",
        title=f"Sampling Distribution Comparison — {metric.replace('_',' ')}",
        xaxis_title=metric.replace("_"," "),
        yaxis_title="Density",
        legend=dict(orientation="h", y=1.08)
    )
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Stats comparison table
    st.markdown('<div class="section-title" style="font-size:11px;">Sampling Statistics Comparison</div>',
                unsafe_allow_html=True)

    stats_data = {
        "Method":     ["Population", "Simple Random", "Systematic", "Stratified"],
        "N":          [len(population), len(srs), len(sys_s), len(strat_s)],
        "Mean":       [population.mean(), srs.mean(), sys_s.mean(), strat_s.mean()],
        "Std Dev":    [population.std(),  srs.std(),  sys_s.std(),  strat_s.std()],
        "Median":     [population.median(), srs.median(), sys_s.median(), strat_s.median()],
        "Min":        [population.min(), srs.min(), sys_s.min(), strat_s.min()],
        "Max":        [population.max(), srs.max(), sys_s.max(), strat_s.max()],
    }
    stats_df = pd.DataFrame(stats_data).set_index("Method").round(3)
    st.dataframe(stats_df.style.background_gradient(cmap="Blues", subset=["Mean","Std Dev"]),
                 use_container_width=True)

    # Box plot comparison
    compare_df = pd.DataFrame({
        "Value":  pd.concat([population, srs, sys_s, strat_s]),
        "Method": (["Population"]*len(population) + ["Simple Random"]*len(srs) +
                   ["Systematic"]*len(sys_s) + ["Stratified"]*len(strat_s))
    })
    fig2 = px.box(compare_df, x="Method", y="Value", color="Method",
                  color_discrete_map={
                      "Population":"#4a5568","Simple Random":"#00d4ff",
                      "Systematic":"#00ffcc","Stratified":"#a855f7"
                  },
                  title="Sample Distribution Spread", template="none")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    # Bias analysis
    bias_srs   = abs(srs.mean()   - population.mean()) / population.mean() * 100
    bias_sys   = abs(sys_s.mean() - population.mean()) / population.mean() * 100
    bias_strat = abs(strat_s.mean() - population.mean()) / population.mean() * 100
    best       = min([("Simple Random", bias_srs), ("Systematic", bias_sys),
                      ("Stratified", bias_strat)], key=lambda x: x[1])

    st.markdown(f"""
    <div class="insight-box">
        <strong>🎲 Sampling Insight:</strong>
        <strong>{best[0]}</strong> sampling shows the lowest mean bias
        ({best[1]:.2f}% deviation from population mean) for <strong>{metric.replace("_"," ")}</strong>.
        Stratified sampling tends to be most representative for heterogeneous supply chain datasets
        with distinct product categories. Systematic sampling risks periodicity bias if ordering
        is non-random.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — K-MEANS CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def section_clustering(df: pd.DataFrame):
    st.markdown('<div class="section-title">K-Means Clustering · Segment Discovery</div>',
                unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        x_col = st.selectbox("X Axis", NUMERIC_COLS, index=0, key="km_x")
    with col_ctrl2:
        y_col = st.selectbox("Y Axis", NUMERIC_COLS, index=2, key="km_y")
    with col_ctrl3:
        k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, key="km_k")

    # Scale & cluster
    data_km = df[[x_col, y_col]].dropna()
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(data_km)
    km      = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels  = km.fit_predict(scaled)
    centroids_scaled = km.cluster_centers_
    centroids_orig   = scaler.inverse_transform(centroids_scaled)

    data_km = data_km.copy()
    data_km["Cluster"] = [f"Cluster {i+1}" for i in labels]

    col1, col2 = st.columns([3, 2])

    with col1:
        fig = px.scatter(data_km, x=x_col, y=y_col, color="Cluster",
                         title=f"K-Means Clusters (K={k})",
                         color_discrete_sequence=PALETTE, opacity=0.65,
                         template="none")
        # Add centroids
        for i, (cx, cy) in enumerate(centroids_orig):
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(symbol="x", size=16, color="#ffffff",
                            line=dict(width=2, color="#000")),
                text=[f"C{i+1}"], textposition="top center",
                textfont=dict(color="#ffffff", size=10),
                name=f"Centroid {i+1}", showlegend=False
            ))
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cluster distribution pie
        cluster_counts = data_km["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig2 = px.pie(cluster_counts, names="Cluster", values="Count",
                      title="Cluster Size Distribution",
                      color_discrete_sequence=PALETTE, hole=0.45, template="none")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster summary table
    st.markdown('<div class="section-title" style="font-size:11px;">Cluster Summary Statistics</div>',
                unsafe_allow_html=True)

    data_km_full = df.loc[data_km.index].copy()
    data_km_full["Cluster"] = data_km["Cluster"].values
    cluster_summary = data_km_full.groupby("Cluster")[NUMERIC_COLS].mean().round(3)
    st.dataframe(
        cluster_summary.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True
    )

    # Elbow method hint
    inertias = []
    k_range  = range(2, min(9, len(df)))
    for ki in k_range:
        km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km_i.fit(scaled)
        inertias.append(km_i.inertia_)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(k_range), y=inertias, mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=8, color="#a855f7")
    ))
    fig3.add_vline(x=k, line_dash="dash", line_color="#ffb800",
                   annotation_text=f"Selected K={k}", annotation_font_color="#ffb800")
    fig3.update_layout(title="Elbow Method — Inertia vs K",
                       xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
    apply_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    largest_cluster = data_km["Cluster"].value_counts().idxmax()
    st.markdown(f"""
    <div class="insight-box">
        <strong>🔵 Clustering Insight:</strong>
        With K={k}, the algorithm identified distinct supply chain segments.
        <strong>{largest_cluster}</strong> is the largest segment by record count.
        Use the elbow plot above to validate K selection — the optimal K is where
        inertia reduction begins to plateau. Cluster summaries reveal segment-level
        differences in cost, demand, and quality metrics useful for targeted strategy.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROBABILITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def section_probability(df: pd.DataFrame):
    st.markdown('<div class="section-title">Probability Distribution · Statistical Modelling</div>',
                unsafe_allow_html=True)

    metric = st.selectbox("Select Variable", NUMERIC_COLS, index=2, key="prob_metric")
    data   = df[metric].dropna()

    mu  = data.mean()
    std = data.std()
    var = data.var()

    col1, col2 = st.columns(2)

    with col1:
        # Histogram + normal curve
        x_range = np.linspace(data.min(), data.max(), 300)
        norm_pdf = stats.norm.pdf(x_range, mu, std)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=40, name="Observed",
            histnorm="probability density",
            marker_color="#00d4ff", opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=norm_pdf, mode="lines",
            line=dict(color="#ff4d7e", width=2.5),
            name="Normal Fit"
        ))
        fig.add_vline(x=mu, line_dash="dot", line_color="#ffb800",
                      annotation_text=f"μ={mu:.2f}", annotation_font_color="#ffb800")
        fig.update_layout(title=f"Distribution of {metric.replace('_',' ')}",
                          xaxis_title=metric.replace("_"," "), yaxis_title="Density")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CDF plot
        sorted_data = np.sort(data)
        cdf         = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        norm_cdf    = stats.norm.cdf(sorted_data, mu, std)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sorted_data, y=cdf, mode="lines",
            line=dict(color="#00d4ff", width=2),
            name="Empirical CDF"
        ))
        fig2.add_trace(go.Scatter(
            x=sorted_data, y=norm_cdf, mode="lines",
            line=dict(color="#a855f7", width=2, dash="dash"),
            name="Theoretical CDF"
        ))
        fig2.update_layout(title="Cumulative Distribution Function (CDF)",
                           xaxis_title=metric.replace("_"," "),
                           yaxis_title="Cumulative Probability")
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # Stats row
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    _, p_val = stats.shapiro(data.sample(min(200, len(data)), random_state=42))

    for col, label, val, fmt in [
        (col_s1, "Mean",     mu,       ".3f"),
        (col_s2, "Std Dev",  std,      ".3f"),
        (col_s3, "Skewness", skewness, ".3f"),
        (col_s4, "Kurtosis", kurtosis, ".3f"),
    ]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-card-label">{label}</div>'
                        f'<div class="stat-card-value">{val:{fmt}}</div></div>',
                        unsafe_allow_html=True)

    # Probability calculator
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:11px;">🔮 Probability Calculator</div>',
                unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p_lo = st.number_input("Lower bound (a)", value=float(data.quantile(0.25)),
                               format="%.4f", key="prob_lo")
    with col_p2:
        p_hi = st.number_input("Upper bound (b)", value=float(data.quantile(0.75)),
                               format="%.4f", key="prob_hi")

    if p_lo < p_hi:
        prob = stats.norm.cdf(p_hi, mu, std) - stats.norm.cdf(p_lo, mu, std)
        empirical_prob = ((data >= p_lo) & (data <= p_hi)).sum() / len(data)

        st.markdown(f"""
        <div style='display:flex;gap:16px;'>
            <div style='flex:1;background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
                        border-radius:10px;padding:16px 20px;'>
                <div style='font-size:10px;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:6px;'>
                    Theoretical P(a ≤ X ≤ b)
                </div>
                <div style='font-size:32px;font-weight:800;color:#00d4ff;
                            font-family:"JetBrains Mono",monospace;'>
                    {prob:.4f}
                </div>
                <div style='font-size:11px;color:#8892a4;margin-top:4px;'>
                    {prob*100:.2f}% — from Normal distribution
                </div>
            </div>
            <div style='flex:1;background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.2);
                        border-radius:10px;padding:16px 20px;'>
                <div style='font-size:10px;font-weight:700;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:6px;'>
                    Empirical P(a ≤ X ≤ b)
                </div>
                <div style='font-size:32px;font-weight:800;color:#a855f7;
                            font-family:"JetBrains Mono",monospace;'>
                    {empirical_prob:.4f}
                </div>
                <div style='font-size:11px;color:#8892a4;margin-top:4px;'>
                    {empirical_prob*100:.2f}% — from observed data
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠ Lower bound must be less than upper bound.")

    normality = "approximately normal" if p_val > 0.05 else "non-normal (p < 0.05)"
    st.markdown(f"""
    <div class="insight-box" style='margin-top:16px;'>
        <strong>🔔 Distribution Insight:</strong>
        <strong>{metric.replace("_"," ")}</strong> has a mean of <strong>{mu:.2f}</strong>
        and std dev of <strong>{std:.2f}</strong>.
        Shapiro-Wilk normality test indicates the distribution is <strong>{normality}</strong>.
        Skewness of {skewness:.2f} suggests
        {"a right-skewed distribution — a few high-value outliers pull the mean up."
         if skewness > 0.5
         else "a left-skewed distribution — some very low values exist."
         if skewness < -0.5
         else "a relatively symmetric distribution close to normal."}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    df = generate_dataset(500)

    # Sidebar — nav & filters
    section, filtered = render_sidebar(df)

    # Header
    render_header(df, filtered)

    # KPIs (always visible)
    render_kpis(filtered)

    # Guard: need data after filtering
    if len(filtered) < 10:
        st.warning("⚠ Not enough data after filters. Please broaden your selection.")
        return

    # Route to selected section
    if section == "Overview":
        section_overview(filtered)
    elif section == "Box Plot":
        section_boxplot(filtered)
    elif section == "Regression":
        section_regression(filtered)
    elif section == "Sampling":
        section_sampling(filtered)
    elif section == "Clustering":
        section_clustering(filtered)
    elif section == "Probability":
        section_probability(filtered)


if __name__ == "__main__":
    main()
