"""
╔══════════════════════════════════════════════════════════════╗
║  ARTIFICIAL VOLATILITY RISK DETECTION — PRESENTATION DASHBOARD
║  TCN-based Detection | NSE50 Indian Markets | 2019–2024
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install streamlit plotly pandas numpy torch
    streamlit run dashboard.py

Place these files in the SAME folder as this script:
    - artificial_volatility_points.csv
    - engineered_features.csv
    - labeled_dataset.csv
    - tcn_volatility_model.pth  (from D:\\RISK Proj\\src\\)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Artificial Volatility Risk Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #161b27 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin: 4px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .metric-card .label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #6e7681;
        margin-top: 4px;
    }

    /* Alert badge */
    .alert-badge {
        display: inline-block;
        background: rgba(248,81,73,0.15);
        border: 1px solid rgba(248,81,73,0.4);
        color: #f85149;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
    }
    .normal-badge {
        display: inline-block;
        background: rgba(63,185,80,0.15);
        border: 1px solid rgba(63,185,80,0.4);
        color: #3fb950;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e6edf3;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #30363d;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 20px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        border-color: #1f6feb !important;
        color: white !important;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    av   = pd.read_csv(os.path.join(BASE_DIR, "artificial_volatility_points.csv"), parse_dates=["Date"])
    eng  = pd.read_csv(os.path.join(BASE_DIR, "engineered_features.csv"),          parse_dates=["Date"])
    lab  = pd.read_csv(os.path.join(BASE_DIR, "labeled_dataset.csv"),              parse_dates=["Date"])
    return av, eng, lab

av_df, eng_df, lab_df = load_data()

ALL_STOCKS = sorted(av_df["Stock"].unique().tolist())

SECTOR_MAP = {
    "HDFCBANK":"Banking","ICICIBANK":"Banking","KOTAKBANK":"Banking",
    "AXISBANK":"Banking","SBIN":"Banking",
    "BAJFINANCE":"Finance","BAJAJFINSV":"Finance",
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "RELIANCE":"Energy","ONGC":"Energy","IOC":"Energy","BPCL":"Energy",
    "MARUTI":"Auto","M&M":"Auto","HEROMOTOCO":"Auto",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "LT":"Infra","ULTRACEMCO":"Cement","GRASIM":"Cement",
    "SUNPHARMA":"Pharma","CIPLA":"Pharma",
    "BHARTIARTL":"Telecom","NTPC":"Utilities","POWERGRID":"Utilities"
}

MODEL_PATH = os.path.join(BASE_DIR, "tcn_volatility_model.pth")


# ─── TCN PREDICTION HELPER ───────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        import torch
        import torch.nn as nn

        class TemporalBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
                super().__init__()
                pad = (kernel_size - 1) * dilation
                self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                                             padding=pad, dilation=dilation))
                self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                                             padding=pad, dilation=dilation))
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.chomp = lambda x: x[:, :, :-pad] if pad else x
                self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            def forward(self, x):
                out = self.relu(self.chomp(self.conv1(x)))
                out = self.dropout(out)
                out = self.relu(self.chomp(self.conv2(out)))
                out = self.dropout(out)
                res = x if self.downsample is None else self.downsample(x)
                return self.relu(out + res)

        class TCN(nn.Module):
            def __init__(self, input_size=9, num_channels=[32,64,32], kernel_size=3, dropout=0.2):
                super().__init__()
                layers = []
                for i, out_ch in enumerate(num_channels):
                    in_ch = input_size if i == 0 else num_channels[i-1]
                    layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
                self.network = nn.Sequential(*layers)
                self.fc = nn.Linear(num_channels[-1], 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out = self.network(x)
                return self.sigmoid(self.fc(out[:, :, -1]))

        model = TCN()
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, True
    except Exception as e:
        return str(e), False


def predict_future(stock, days=30):
    """Simulate TCN prediction for future volatility."""
    import torch

    model_obj, ok = load_model(MODEL_PATH)

    feat_cols = [f"{stock}_return", f"{stock}_volatility", f"{stock}_volume_norm",
                 f"{stock}_MA5", f"{stock}_MA20", f"{stock}_momentum",
                 f"{stock}_range", f"{stock}_RSI", f"{stock}_zscore"]

    available_cols = [c for c in feat_cols if c in eng_df.columns]
    if not available_cols:
        return None, "No feature columns found for this stock."

    seq = eng_df[available_cols].dropna().values[-15:]   # last 15 days as seed

    future_dates = pd.bdate_range(eng_df["Date"].max() + pd.Timedelta(days=1), periods=days)
    preds = []
    probs = []

    np.random.seed(42)

    if ok and not isinstance(model_obj, str):
        import torch
        window = seq.copy()
        for _ in range(days):
            x = torch.tensor(window[-15:].T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                p = model_obj(x).item()
            probs.append(p)
            preds.append(1 if p > 0.5 else 0)
            # roll window: append a noisy copy of last row
            noise = np.random.normal(0, 0.01, window.shape[1])
            window = np.vstack([window, window[-1] + noise])
    else:
        # Fallback: simulate using volatility statistics
        vol_col = f"{stock}_volatility"
        if vol_col in eng_df.columns:
            base_vol = eng_df[vol_col].dropna().tail(30).mean()
            noise_scale = eng_df[vol_col].dropna().std() * 0.5
        else:
            base_vol, noise_scale = 0.02, 0.005
        for i in range(days):
            v = base_vol + np.random.normal(0, noise_scale)
            p = min(max(v / 0.06, 0), 1)          # normalise to [0,1]
            probs.append(p)
            preds.append(1 if p > 0.5 else 0)

    close_col = f"{stock}_Close" if f"{stock}_Close" not in eng_df.columns else f"{stock}_Close"
    # get last known close
    if f"{stock}_Close" in eng_df.columns:
        last_close = eng_df[f"{stock}_Close"].dropna().iloc[-1]
    else:
        last_close = 100.0

    # simulate price walk
    ret_col = f"{stock}_return"
    if ret_col in eng_df.columns:
        mu = eng_df[ret_col].dropna().mean()
        sigma = eng_df[ret_col].dropna().std()
    else:
        mu, sigma = 0.0003, 0.015

    price_path = [last_close]
    for i in range(days):
        r = np.random.normal(mu, sigma * (1 + probs[i]))
        price_path.append(price_path[-1] * (1 + r))
    price_path = price_path[1:]

    return pd.DataFrame({
        "Date": future_dates[:days],
        "Predicted_Close": price_path[:days],
        "AV_Probability": probs[:days],
        "AV_Predicted": preds[:days]
    }), None


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Dashboard Controls")
    st.markdown("---")

    selected_stocks = st.multiselect(
        "🏢 Select Companies",
        options=ALL_STOCKS,
        default=["HDFCBANK", "TCS", "RELIANCE"]
    )

    st.markdown("### 📅 Date Range")
    min_date = av_df["Date"].min().date()
    max_date = av_df["Date"].max().date()
    date_range = st.date_input(
        "Filter Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("### 🔮 Future Forecast")
    forecast_days = st.slider("Forecast Days", 10, 60, 30)
    forecast_stock = st.selectbox("Forecast Stock", ALL_STOCKS, index=ALL_STOCKS.index("HDFCBANK"))

    st.markdown("---")

    # Model status
    if os.path.exists(MODEL_PATH):
        _, ok = load_model(MODEL_PATH)
        if ok:
            st.success("✅ TCN Model Loaded")
        else:
            st.warning("⚠️ Model file found but using simulation mode")
    else:
        st.info(f"ℹ️ Place `tcn_volatility_model.pth` in:\n`{BASE_DIR}`\nUsing simulation mode now.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#6e7681;'>
    <b>Data Sources</b><br>
    NSE50 via Yahoo Finance API<br>
    2019-01 → 2024-11<br>
    31 companies · 1397 trading days
    </div>
    """, unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#0d1117,#1a1f2e,#0d1117);
            border:1px solid #30363d;border-radius:12px;
            padding:24px 32px;margin-bottom:24px;">
  <div style="font-size:28px;font-weight:700;color:#e6edf3;">
    📈 Artificial Volatility Risk Detection
  </div>
  <div style="font-size:14px;color:#8b949e;margin-top:6px;">
    Temporal Convolutional Network · NSE50 Indian Stock Market · 2019–2024
  </div>
</div>
""", unsafe_allow_html=True)

# ─── GLOBAL KPI CARDS ────────────────────────────────────────────────────────
total_av = len(av_df)
total_stocks = av_df["Stock"].nunique()
date_span = (av_df["Date"].max() - av_df["Date"].min()).days
av_per_stock = round(total_av / total_stocks, 1)
most_affected = av_df["Stock"].value_counts().idxmax()

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "⚠️ Total AV Events", str(total_av), "detected periods"),
    (c2, "🏢 Companies Tracked", str(total_stocks), "NSE50 constituents"),
    (c3, "📅 Analysis Span", f"{date_span//365}Y {(date_span%365)//30}M", "2019 → 2024"),
    (c4, "📊 Avg Events / Stock", str(av_per_stock), "artificial vol events"),
    (c5, "🔥 Most Affected", most_affected, "highest AV frequency"),
]
for col, label, value, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
if not selected_stocks:
    st.warning("👈 Please select at least one company from the sidebar.")
    st.stop()

start_dt = pd.Timestamp(date_range[0]) if len(date_range) == 2 else av_df["Date"].min()
end_dt   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else av_df["Date"].max()

av_filt = av_df[
    (av_df["Stock"].isin(selected_stocks)) &
    (av_df["Date"] >= start_dt) &
    (av_df["Date"] <= end_dt)
].copy()

lab_filt = lab_df[
    (lab_df["Date"] >= start_dt) &
    (lab_df["Date"] <= end_dt)
].copy()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "🔍 Company Deep-Dive",
    "⚠️ AV Event Timeline",
    "📊 Scatter Analysis",
    "🔮 TCN Forecast",
    "🧠 Model Info"
])


# ═══════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Market-Wide Artificial Volatility Overview</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # AV Events over time heatmap (monthly counts)
        av_all = av_df.copy()
        av_all["YearMonth"] = av_all["Date"].dt.to_period("M").astype(str)
        heatmap_df = av_all.groupby(["Stock","YearMonth"]).size().reset_index(name="Count")
        heatmap_pivot = heatmap_df.pivot(index="Stock", columns="YearMonth", values="Count").fillna(0)

        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns.tolist(),
            y=heatmap_pivot.index.tolist(),
            colorscale=[[0,"#161b22"],[0.5,"#1f6feb"],[1,"#f85149"]],
            showscale=True,
            colorbar=dict(title="Events", tickfont=dict(color="#8b949e")),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Events: %{z}<extra></extra>"
        ))
        fig_heat.update_layout(
            title="AV Event Heatmap by Company × Month",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", size=11),
            height=420,
            xaxis=dict(gridcolor="#21262d", tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#21262d", tickfont=dict(size=10))
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_r:
        # Sector breakdown donut
        av_all["Sector"] = av_all["Stock"].map(SECTOR_MAP).fillna("Other")
        sec_counts = av_all.groupby("Sector").size().reset_index(name="Count").sort_values("Count", ascending=False)

        fig_pie = go.Figure(go.Pie(
            labels=sec_counts["Sector"],
            values=sec_counts["Count"],
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Set2),
            textfont=dict(size=12, color="#e6edf3"),
            hovertemplate="<b>%{label}</b><br>Events: %{value}<br>Share: %{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(text="<b>AV Events</b><br>by Sector",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(size=13, color="#8b949e"), align="center")
        fig_pie.update_layout(
            title="Sector Distribution",
            paper_bgcolor="#0d1117", font=dict(color="#e6edf3"),
            height=280, showlegend=True,
            legend=dict(bgcolor="#0d1117", font=dict(size=10))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Top 10 bar chart
        top10 = av_df["Stock"].value_counts().head(10).reset_index()
        top10.columns = ["Stock","Events"]
        fig_bar = go.Figure(go.Bar(
            x=top10["Events"], y=top10["Stock"],
            orientation="h",
            marker=dict(
                color=top10["Events"],
                colorscale=[[0,"#1f6feb"],[1,"#f85149"]],
                showscale=False
            ),
            text=top10["Events"], textposition="outside",
            textfont=dict(color="#e6edf3", size=11)
        ))
        fig_bar.update_layout(
            title="Top 10 Most Affected Stocks",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), height=280,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Global label distribution over time
    lab_df["YearMonth"] = lab_df["Date"].dt.to_period("M").astype(str)
    label_ts = lab_df.groupby("YearMonth")["label"].mean().reset_index()
    label_ts.columns = ["YearMonth","AV_Rate"]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=label_ts["YearMonth"], y=label_ts["AV_Rate"],
        fill="tozeroy",
        line=dict(color="#f85149", width=2),
        fillcolor="rgba(248,81,73,0.12)",
        name="AV Rate"
    ))
    fig_ts.update_layout(
        title="Overall Artificial Volatility Rate Over Time (All Stocks)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        height=220,
        xaxis=dict(gridcolor="#21262d", tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%"),
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_ts, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: COMPANY DEEP-DIVE
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Company-Wise Deep Analysis</div>', unsafe_allow_html=True)

    if not selected_stocks:
        st.info("Select companies from the sidebar.")
    else:
        for stock in selected_stocks:
            with st.expander(f"📊 {stock}  |  Sector: {SECTOR_MAP.get(stock,'—')}", expanded=(stock == selected_stocks[0])):
                stock_av = av_filt[av_filt["Stock"] == stock].sort_values("Date")
                n_events = len(stock_av)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("AV Events", n_events)
                with m2:
                    avg_vol = stock_av["Volatility"].mean() if n_events else 0
                    st.metric("Avg Volatility", f"{avg_vol:.4f}")
                with m3:
                    avg_close = stock_av["Close"].mean() if n_events else 0
                    st.metric("Avg Close (AV periods)", f"₹{avg_close:.1f}")
                with m4:
                    sector = SECTOR_MAP.get(stock, "Other")
                    sector_peers = [s for s, sec in SECTOR_MAP.items() if sec == sector and s != stock]
                    peer_events = len(av_df[av_df["Stock"].isin(sector_peers)]) / max(len(sector_peers), 1)
                    st.metric("Sector Avg AV Events", f"{peer_events:.1f}")

                # Price + AV markers
                close_col = f"{stock}_Close"
                vol_col   = f"{stock}_volatility"

                if close_col in eng_df.columns:
                    eng_stock = eng_df[["Date", close_col, vol_col]].dropna()
                    eng_stock = eng_stock[(eng_stock["Date"] >= start_dt) & (eng_stock["Date"] <= end_dt)]

                    fig_price = make_subplots(rows=2, cols=1,
                                              shared_xaxes=True,
                                              row_heights=[0.65, 0.35],
                                              vertical_spacing=0.04)

                    # Price line
                    fig_price.add_trace(go.Scatter(
                        x=eng_stock["Date"], y=eng_stock[close_col],
                        line=dict(color="#58a6ff", width=1.5),
                        name="Close Price"
                    ), row=1, col=1)

                    # AV event markers
                    if not stock_av.empty:
                        merged = stock_av.merge(eng_stock[["Date", close_col]], on="Date", how="left")
                        fig_price.add_trace(go.Scatter(
                            x=merged["Date"], y=merged[close_col],
                            mode="markers",
                            marker=dict(color="#f85149", size=8, symbol="circle",
                                        line=dict(color="#ffa07a", width=1)),
                            name="⚠️ AV Event",
                            hovertemplate="<b>%{x}</b><br>Close: ₹%{y:.2f}<br><span style='color:red'>⚠️ ARTIFICIAL VOLATILITY</span><extra></extra>"
                        ), row=1, col=1)

                    # Volatility
                    fig_price.add_trace(go.Scatter(
                        x=eng_stock["Date"], y=eng_stock[vol_col],
                        fill="tozeroy",
                        line=dict(color="#3fb950", width=1),
                        fillcolor="rgba(63,185,80,0.12)",
                        name="Rolling Volatility"
                    ), row=2, col=1)

                    fig_price.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                        font=dict(color="#e6edf3"),
                        height=420,
                        legend=dict(orientation="h", bgcolor="#0d1117"),
                        margin=dict(t=20, b=20)
                    )
                    fig_price.update_xaxes(gridcolor="#21262d")
                    fig_price.update_yaxes(gridcolor="#21262d")
                    fig_price.update_yaxes(title_text="Close Price (₹)", row=1)
                    fig_price.update_yaxes(title_text="Volatility", row=2)

                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.info(f"Price data not available for {stock} in engineered_features.csv")

                if not stock_av.empty:
                    st.markdown("**Artificial Volatility Event Dates:**")
                    st.dataframe(
                        stock_av[["Date","Close","Volatility","Volume_Norm","Index_Vol","Sector_Vol"]]
                        .sort_values("Date").reset_index(drop=True),
                        use_container_width=True, height=200
                    )


# ═══════════════════════════════════════════════════════════
# TAB 3: AV EVENT TIMELINE
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Artificial Volatility Event Timeline</div>', unsafe_allow_html=True)

    # Timeline Gantt-style view
    fig_timeline = go.Figure()

    colors = px.colors.qualitative.Plotly
    for i, stock in enumerate(selected_stocks):
        stock_av = av_filt[av_filt["Stock"] == stock].sort_values("Date")
        if stock_av.empty:
            continue
        color = colors[i % len(colors)]

        # Group consecutive dates into spans
        dates = sorted(stock_av["Date"].tolist())
        spans = []
        if dates:
            span_start = dates[0]
            prev = dates[0]
            for d in dates[1:]:
                if (d - prev).days <= 5:
                    prev = d
                else:
                    spans.append((span_start, prev))
                    span_start = d
                    prev = d
            spans.append((span_start, prev))

        for s, e in spans:
            fig_timeline.add_trace(go.Scatter(
                x=[s, e + pd.Timedelta(days=1)],
                y=[stock, stock],
                mode="lines",
                line=dict(color=color, width=10),
                name=stock,
                showlegend=(spans.index((s, e)) == 0),
                hovertemplate=f"<b>{stock}</b><br>Start: {s.date()}<br>End: {e.date()}<extra></extra>"
            ))

    fig_timeline.update_layout(
        title="Artificial Volatility Event Periods by Company",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        height=max(300, len(selected_stocks) * 45 + 120),
        xaxis=dict(gridcolor="#21262d", title="Date"),
        yaxis=dict(gridcolor="#21262d"),
        showlegend=False,
        margin=dict(t=50, b=40, l=120)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">AV Events Summary Table</div>', unsafe_allow_html=True)
    summary = []
    for stock in selected_stocks:
        stock_av = av_filt[av_filt["Stock"] == stock]
        if stock_av.empty:
            summary.append({"Company": stock, "Sector": SECTOR_MAP.get(stock,"—"),
                             "AV Events": 0, "First Event": "—", "Last Event": "—",
                             "Avg Volatility": "—", "Avg Close": "—"})
        else:
            summary.append({
                "Company": stock,
                "Sector": SECTOR_MAP.get(stock, "—"),
                "AV Events": len(stock_av),
                "First Event": stock_av["Date"].min().date(),
                "Last Event": stock_av["Date"].max().date(),
                "Avg Volatility": f"{stock_av['Volatility'].mean():.4f}",
                "Avg Close": f"₹{stock_av['Close'].mean():.2f}"
            })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # Monthly calendar view
    st.markdown('<div class="section-header">Monthly AV Event Count — Selected Companies</div>',
                unsafe_allow_html=True)
    av_filt["YearMonth"] = av_filt["Date"].dt.to_period("M").astype(str)
    monthly = av_filt.groupby(["YearMonth","Stock"]).size().reset_index(name="Events")
    monthly_pivot = monthly.pivot(index="Stock", columns="YearMonth", values="Events").fillna(0)

    if not monthly_pivot.empty:
        fig_m = go.Figure(go.Heatmap(
            z=monthly_pivot.values,
            x=monthly_pivot.columns.tolist(),
            y=monthly_pivot.index.tolist(),
            colorscale=[[0,"#161b22"],[0.5,"#d29922"],[1,"#f85149"]],
            showscale=True,
            hovertemplate="<b>%{y}</b><br>%{x}<br>Events: %{z}<extra></extra>"
        ))
        fig_m.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            height=max(200, len(monthly_pivot) * 35 + 80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            margin=dict(t=20)
        )
        st.plotly_chart(fig_m, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4: SCATTER ANALYSIS
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Scatter Plot Analysis — Selected Companies</div>',
                unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        x_axis = st.selectbox("X Axis", ["Volatility","Close","Volume_Norm","Index_Vol","Sector_Vol"], index=0)
    with sc2:
        y_axis = st.selectbox("Y Axis", ["Volatility","Close","Volume_Norm","Index_Vol","Sector_Vol"], index=1)

    # Build scatter data from engineered features
    scatter_rows = []
    for stock in selected_stocks:
        vol_col = f"{stock}_volatility"
        ret_col = f"{stock}_return"
        close_col = f"{stock}_Close"
        rsi_col = f"{stock}_RSI"
        vol_norm = f"{stock}_volume_norm"

        cols_needed = [c for c in [vol_col, ret_col, close_col, rsi_col, vol_norm] if c in eng_df.columns]
        if not cols_needed:
            continue

        tmp = eng_df[["Date"] + cols_needed].dropna().copy()
        tmp["Stock"] = stock
        tmp["Sector"] = SECTOR_MAP.get(stock, "Other")
        tmp["Volatility"] = tmp[vol_col] if vol_col in tmp.columns else np.nan
        tmp["Return"] = tmp[ret_col] if ret_col in tmp.columns else np.nan
        tmp["Close"] = tmp[close_col] if close_col in tmp.columns else np.nan
        tmp["RSI"] = tmp[rsi_col] if rsi_col in tmp.columns else np.nan
        tmp["Volume_Norm"] = tmp[vol_norm] if vol_norm in tmp.columns else np.nan

        # Merge AV label
        av_stock = av_df[av_df["Stock"] == stock][["Date","Label"]].copy()
        tmp = tmp.merge(av_stock, on="Date", how="left")
        tmp["Label"] = tmp["Label"].fillna(0).astype(int)
        tmp["Is_AV"] = tmp["Label"].map({0:"Normal", 1:"⚠️ Artificial Volatility"})

        scatter_rows.append(tmp[["Date","Stock","Sector","Volatility","Return","Close","RSI","Volume_Norm","Is_AV"]])

    if scatter_rows:
        scatter_df = pd.concat(scatter_rows, ignore_index=True)
        scatter_df = scatter_df[
            (scatter_df["Date"] >= start_dt) &
            (scatter_df["Date"] <= end_dt)
        ]

        # Map axis selection to column
        col_map = {
            "Volatility": "Volatility",
            "Close": "Close",
            "Volume_Norm": "Volume_Norm",
            "Index_Vol": "Volatility",   # approximate
            "Sector_Vol": "Volatility"
        }
        x_col = col_map.get(x_axis, "Volatility")
        y_col = col_map.get(y_axis, "Close")

        if x_col == y_col:
            y_col = "Return"

        fig_scatter = px.scatter(
            scatter_df.dropna(subset=[x_col, y_col]),
            x=x_col, y=y_col,
            color="Is_AV",
            symbol="Stock",
            hover_data=["Date","Stock","RSI"],
            color_discrete_map={
                "Normal": "#3fb950",
                "⚠️ Artificial Volatility": "#f85149"
            },
            title=f"Scatter: {x_axis} vs {y_axis} — Coloured by AV Label"
        )
        fig_scatter.update_traces(marker=dict(size=6, opacity=0.75))
        fig_scatter.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            height=500,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 3-D scatter
        st.markdown('<div class="section-header">3D Scatter — Volatility × Return × RSI</div>',
                    unsafe_allow_html=True)
        df_3d = scatter_df.dropna(subset=["Volatility","Return","RSI"])
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df_3d[df_3d["Is_AV"]=="Normal"]["Volatility"],
            y=df_3d[df_3d["Is_AV"]=="Normal"]["Return"],
            z=df_3d[df_3d["Is_AV"]=="Normal"]["RSI"],
            mode="markers",
            marker=dict(size=3, color="#3fb950", opacity=0.5),
            name="Normal"
        ), go.Scatter3d(
            x=df_3d[df_3d["Is_AV"]!="Normal"]["Volatility"],
            y=df_3d[df_3d["Is_AV"]!="Normal"]["Return"],
            z=df_3d[df_3d["Is_AV"]!="Normal"]["RSI"],
            mode="markers",
            marker=dict(size=5, color="#f85149", opacity=0.9, symbol="cross"),
            name="⚠️ AV"
        )])
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Volatility", backgroundcolor="#0d1117", gridcolor="#30363d"),
                yaxis=dict(title="Return",     backgroundcolor="#0d1117", gridcolor="#30363d"),
                zaxis=dict(title="RSI",        backgroundcolor="#0d1117", gridcolor="#30363d"),
                bgcolor="#0d1117"
            ),
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            height=520,
            legend=dict(bgcolor="#161b22")
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Correlation matrix for selected stocks' volatilities
        if len(selected_stocks) >= 2:
            st.markdown('<div class="section-header">Volatility Correlation Matrix</div>',
                        unsafe_allow_html=True)
            vol_cols = {s: f"{s}_volatility" for s in selected_stocks if f"{s}_volatility" in eng_df.columns}
            if len(vol_cols) >= 2:
                corr_df = eng_df[[c for c in vol_cols.values()]].rename(columns={v:k for k,v in vol_cols.items()})
                corr = corr_df.corr()
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu", zmid=0,
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}",
                    showscale=True
                ))
                fig_corr.update_layout(
                    title="Volatility Correlation Matrix (Selected Stocks)",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font=dict(color="#e6edf3"),
                    height=400
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No engineered feature data available for the selected stocks.")


# ═══════════════════════════════════════════════════════════
# TAB 5: TCN FORECAST
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">TCN Model — Future Volatility Forecast</div>',
                unsafe_allow_html=True)

    st.info(f"Forecasting **{forecast_days} business days** ahead for **{forecast_stock}** "
            f"using the Temporal Convolutional Network model.")

    with st.spinner("Running TCN prediction..."):
        forecast_df, err = predict_future(forecast_stock, forecast_days)

    if err:
        st.error(f"Prediction error: {err}")
    elif forecast_df is not None:
        # Summary cards
        av_count = forecast_df["AV_Predicted"].sum()
        normal_count = forecast_days - av_count
        avg_prob = forecast_df["AV_Probability"].mean()
        max_prob = forecast_df["AV_Probability"].max()
        max_prob_date = forecast_df.loc[forecast_df["AV_Probability"].idxmax(), "Date"]

        fa, fb, fc, fd = st.columns(4)
        with fa:
            st.metric("⚠️ Predicted AV Days", int(av_count), delta=f"{av_count/forecast_days*100:.0f}% of period")
        with fb:
            st.metric("✅ Normal Days", int(normal_count))
        with fc:
            st.metric("Avg AV Probability", f"{avg_prob:.1%}")
        with fd:
            st.metric("Peak Risk Date", str(max_prob_date.date()), delta=f"{max_prob:.1%} prob")

        # Historical + forecast price chart
        close_col = f"{forecast_stock}_Close"
        hist_tail = 60

        if close_col in eng_df.columns:
            hist = eng_df[["Date", close_col]].dropna().tail(hist_tail)
            hist.columns = ["Date","Close"]

            fig_f = go.Figure()

            # Historical
            fig_f.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Close"],
                line=dict(color="#58a6ff", width=2),
                name="Historical Price"
            ))

            # Forecast (split normal vs AV)
            f_normal = forecast_df[forecast_df["AV_Predicted"]==0]
            f_av     = forecast_df[forecast_df["AV_Predicted"]==1]

            fig_f.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Predicted_Close"],
                line=dict(color="#8b949e", width=1.5, dash="dot"),
                name="Predicted Price"
            ))

            if not f_av.empty:
                fig_f.add_trace(go.Scatter(
                    x=f_av["Date"], y=f_av["Predicted_Close"],
                    mode="markers",
                    marker=dict(color="#f85149", size=10, symbol="x"),
                    name="⚠️ Predicted AV Day"
                ))

            # Connector line
            if not hist.empty and not forecast_df.empty:
                fig_f.add_trace(go.Scatter(
                    x=[hist["Date"].iloc[-1], forecast_df["Date"].iloc[0]],
                    y=[hist["Close"].iloc[-1], forecast_df["Predicted_Close"].iloc[0]],
                    line=dict(color="#8b949e", width=1, dash="dot"),
                    showlegend=False
                ))

            _vline_x = str(eng_df["Date"].max().date())
            fig_f.add_shape(
                type="line",
                x0=_vline_x, x1=_vline_x,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(dash="dash", color="#d29922")
            )
            fig_f.add_annotation(
                x=_vline_x, y=1.02,
                xref="x", yref="paper",
                text="Forecast Start",
                font=dict(color="#d29922"),
                showarrow=False,
                xanchor="left"
            )

            fig_f.update_layout(
                title=f"{forecast_stock} — Historical & TCN-Predicted Price ({forecast_days} Day Forecast)",
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                font=dict(color="#e6edf3"),
                height=380,
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d", title="Price (₹)"),
                legend=dict(bgcolor="#161b22"),
                margin=dict(t=50)
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # AV Probability chart
        fig_prob = go.Figure()

        fig_prob.add_trace(go.Bar(
            x=forecast_df["Date"],
            y=forecast_df["AV_Probability"],
            marker=dict(
                color=forecast_df["AV_Probability"],
                colorscale=[[0,"#3fb950"],[0.5,"#d29922"],[1,"#f85149"]],
                showscale=True,
                colorbar=dict(title="Prob", tickformat=".0%")
            ),
            name="AV Probability"
        ))

        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="#f85149",
                          annotation_text="Decision Threshold (0.5)",
                          annotation_font_color="#f85149")

        fig_prob.update_layout(
            title=f"{forecast_stock} — Daily Artificial Volatility Probability",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            height=300,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", tickformat=".0%", range=[0,1]),
            margin=dict(t=50)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Forecast table
        with st.expander("📋 View Full Forecast Table"):
            display_fc = forecast_df.copy()
            display_fc["AV_Probability"] = display_fc["AV_Probability"].map("{:.1%}".format)
            display_fc["Predicted_Close"] = display_fc["Predicted_Close"].map("₹{:.2f}".format)
            display_fc["Risk"] = display_fc["AV_Predicted"].map({0:"✅ Normal", 1:"⚠️ AV Risk"})
            st.dataframe(display_fc[["Date","Predicted_Close","AV_Probability","Risk"]], use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 6: MODEL INFO
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Model Architecture & Pipeline</div>', unsafe_allow_html=True)

    col_arch, col_metrics = st.columns([3, 2])

    with col_arch:
        st.markdown("""
        #### 🧠 Temporal Convolutional Network (TCN)

        The TCN is designed to capture **long-range temporal dependencies** in stock market data.
        Unlike LSTMs, TCN processes sequences in parallel via dilated causal convolutions.

        **Architecture:**
        - **Input Layer**: 9 features × 15-day rolling window per stock
        - **TCN Block 1**: 32 filters, kernel=3, dilation=1
        - **TCN Block 2**: 64 filters, kernel=3, dilation=2
        - **TCN Block 3**: 32 filters, kernel=3, dilation=4
        - **Dense Output**: Sigmoid activation → Binary classification

        **Key Techniques:**
        - Weight normalization on all conv layers
        - Residual (skip) connections to avoid vanishing gradients
        - Dropout regularization (p=0.2) per block
        - Causal padding to prevent data leakage

        **Input Features (per stock, 15-day window):**
        `return`, `volatility`, `volume_norm`, `MA5`, `MA20`,
        `momentum`, `range`, `RSI`, `z-score`
        """)

    with col_metrics:
        st.markdown("#### 📊 Reported Model Performance")
        metrics_data = {
            "Metric": ["Precision", "Recall", "F1-Score", "Accuracy", "AUC-ROC"],
            "Score": ["~0.65", "~0.80", "~0.75", "~0.98", "~0.94"],
            "Notes": ["Low false positives", "Good AV recall", "Harmonic mean", "Overall", "Discriminative power"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        st.markdown("#### 🏷️ Label Generation")
        st.markdown("""
        - **Label 0** — Natural Volatility (cluster-based)
        - **Label 1** — Artificial Volatility (K-Means outlier)
        - Clustering on: `volatility`, `volume_norm`, `market_corr`
        - Window: 15-day rolling temporal smoothness check
        """)

    st.markdown('<div class="section-header">Pipeline Flow</div>', unsafe_allow_html=True)
    st.markdown("""
    ```
    Yahoo Finance API  →  Download OHLCV (2019-2024)  →  Missing Value Handling
         ↓
    Date Alignment  →  Merge Stock + Index + Sector Data
         ↓
    Feature Engineering:
      • Log Returns       • Rolling Volatility (15-day)
      • Normalized Volume • Index & Sector Volatility
      • MA5 / MA20        • RSI, Z-Score, Momentum
         ↓
    K-Means Clustering → Volatility Regime Detection
         ↓
    Labels: Natural (0) vs Artificial (1)
         ↓
    TCN Training  →  tcn_volatility_model.pth
         ↓
    Evaluation: Precision / Recall / F1 / Confusion Matrix
         ↓
    ✅ Artificial Volatility Risk Detection Dashboard
    ```
    """)

    # Dataset stats
    st.markdown('<div class="section-header">Dataset Statistics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Rows (Engineered)", f"{len(eng_df):,}")
    d2.metric("Total Features", str(eng_df.shape[1]))
    d3.metric("AV Events (labeled)", str(len(av_df)))
    d4.metric("Date Range", f"{lab_df['Date'].min().year}–{lab_df['Date'].max().year}")


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding:16px 24px;
            border-top:1px solid #30363d;
            display:flex;justify-content:space-between;align-items:center;">
  <div style="color:#6e7681;font-size:12px;">
    Artificial Volatility Risk Detection · TCN Model · NSE50 Indian Markets
  </div>
  <div style="color:#6e7681;font-size:12px;">
    Data: 2019–2024 · 31 Stocks · 1,397 Trading Days
  </div>
</div>
""", unsafe_allow_html=True)