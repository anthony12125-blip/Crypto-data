"""
Monitoring Dashboard
=====================
Real-time view of everything the bot is doing.
Run: streamlit run dashboard.py

Shows:
- BTC momentum gauge (is Bitcoin hot?)
- Market regime indicator
- Top alt amplifiers right now
- Open positions & P/L
- Historical performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import sys
import time

# Page config
st.set_page_config(
    page_title="🚀 Crypto Alpha Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {font-size: 3em; font-weight: bold; text-align: center;}
    .regime-bear {color: #ff4444; font-size: 2em; text-align: center;}
    .regime-bull {color: #00cc66; font-size: 2em; text-align: center;}
    .regime-fire {color: #ff8800; font-size: 2em; text-align: center;}
    .stMetric > div {text-align: center;}
</style>
""", unsafe_allow_html=True)


def get_btc_momentum():
    """Get live BTC momentum."""
    try:
        from btc_momentum import BTCMomentumScanner
        scanner = BTCMomentumScanner("binance")
        return scanner.get_momentum_signal()
    except Exception as e:
        st.error(f"BTC Scanner error: {e}")
        return None


def get_regime():
    """Get current regime."""
    try:
        from regime_detector import RegimeDetector
        detector = RegimeDetector("binance")
        return detector.detect()
    except Exception as e:
        st.error(f"Regime Detector error: {e}")
        return None


def get_alt_scan(btc_momentum=50):
    """Get alt scan results."""
    try:
        from alt_scanner import AltBetaScanner
        scanner = AltBetaScanner("binance")
        return scanner.scan(btc_momentum_score=btc_momentum, top_n_alts=50)
    except Exception as e:
        st.error(f"Alt Scanner error: {e}")
        return []


def get_db_stats():
    """Get collection stats from database."""
    db_path = "data/market_data.db"
    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        stats = {}
        for table in ["ohlcv", "alt_snapshots", "fear_greed"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        conn.close()
        return stats
    except:
        return None


def get_trading_state():
    """Load trading state if available."""
    state_path = Path("logs/state.json")
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None


def create_momentum_gauge(score: float, regime: str) -> go.Figure:
    """Create a speedometer-style gauge for BTC momentum."""
    colors = {
        "dead": "#666666",
        "cool": "#4488ff",
        "warm": "#ffaa00",
        "hot": "#ff6600",
        "fire": "#ff0000",
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"BTC Momentum [{regime.upper()}]", "font": {"size": 24}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": colors.get(regime, "#666")},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 25], "color": "#1a1a2e"},
                {"range": [25, 45], "color": "#16213e"},
                {"range": [45, 65], "color": "#0f3460"},
                {"range": [65, 80], "color": "#e94560"},
                {"range": [80, 100], "color": "#ff0000"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Crypto Alpha Bot")
    st.caption("Alt Beta Amplification System")
    st.markdown("---")

    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.markdown("---")
    st.markdown("### Data Collection")
    db_stats = get_db_stats()
    if db_stats:
        st.success("✅ Collector Active")
        st.metric("OHLCV Records", f"{db_stats.get('ohlcv', 0):,}")
        st.metric("Alt Snapshots", f"{db_stats.get('alt_snapshots', 0):,}")
        st.metric("F&G Records", f"{db_stats.get('fear_greed', 0):,}")
    else:
        st.warning("⚠️ No data yet")
        st.info("Run: `python3 data_collector.py`")

    st.markdown("---")
    st.markdown("### Quick Actions")
    st.code("python3 data_collector.py", language="bash")
    st.code("python3 backtest.py", language="bash")
    st.code("python3 train_models.py", language="bash")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.title("🚀 Crypto Alpha Bot Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Row 1: BTC Momentum + Regime
col1, col2 = st.columns([2, 1])

with col1:
    with st.spinner("Scanning BTC momentum..."):
        btc_signal = get_btc_momentum()

    if btc_signal:
        fig = create_momentum_gauge(btc_signal.score, btc_signal.regime)
        st.plotly_chart(fig, use_container_width=True)

        # BTC details
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("BTC Price", f"${btc_signal.btc_price:,.0f}", f"{btc_signal.btc_change_1h:+.2f}% 1h")
        m2.metric("24h Change", f"{btc_signal.btc_change_24h:+.2f}%")
        m3.metric("Volume", f"{btc_signal.volume_ratio:.1f}x avg")
        m4.metric("RSI (1h)", f"{btc_signal.rsi_1h:.0f}")

        # Score breakdown
        with st.expander("Score Breakdown"):
            for k, v in btc_signal.details.items():
                st.progress(v / 25, text=f"{k}: {v:.1f}/25")

with col2:
    st.subheader("Market Regime")
    with st.spinner("Detecting regime..."):
        regime = get_regime()

    if regime:
        regime_colors = {
            "bear": "🐻 BEAR", "transition": "🌅 TRANSITION",
            "early_bull": "🐂 EARLY BULL", "bull": "🚀 BULL",
            "overheated": "🌋 OVERHEATED"
        }
        st.markdown(f"### {regime_colors.get(regime.regime, regime.regime)}")
        st.metric("Confidence", f"{regime.confidence:.0%}")
        st.metric("vs 200d MA", f"{regime.btc_vs_200d_ma:+.1f}%")
        st.metric("Fear & Greed", f"{regime.fear_greed}/100")
        st.metric("Weekly Trend", regime.weekly_trend.upper())

        if regime.should_trade_alts:
            st.success("🟢 Alt trading ACTIVE")
        elif regime.should_accumulate:
            st.info("🔵 Accumulation mode")
        elif regime.should_exit:
            st.error("🔴 Exit mode — take profits")

st.markdown("---")

# Row 2: Alt Scanner
st.subheader("🔍 Alt Beta Amplification Scanner")

with st.spinner("Scanning top 50 alts... (this takes ~2 minutes)"):
    momentum_score = btc_signal.score if btc_signal else 50
    alt_signals = get_alt_scan(momentum_score)

if alt_signals:
    # Top alts table
    top_data = []
    for i, s in enumerate(alt_signals[:25], 1):
        top_data.append({
            "Rank": i,
            "Symbol": s.symbol.replace("/USDT", ""),
            "Score": s.score,
            "Beta": f"{s.beta_vs_btc:.1f}x",
            "Amplification": f"{s.amplification_ratio:.1f}x",
            "24h %": f"{s.change_24h:+.1f}%",
            "BTC 24h %": f"{s.btc_change_24h:+.1f}%",
            "Vol Surge": f"{s.volume_surge:.1f}x",
            "RSI": f"{s.rsi:.0f}",
            "Correlation": f"{s.correlation_btc:.2f}",
            "Grade": s.grade,
        })

    df_display = pd.DataFrame(top_data)
    st.dataframe(df_display, use_container_width=True, height=600)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Beta distribution
        betas = [s.beta_vs_btc for s in alt_signals[:25]]
        names = [s.symbol.replace("/USDT", "") for s in alt_signals[:25]]
        fig = px.bar(x=names, y=betas, title="Beta vs BTC (Top 25 Alts)",
                     labels={"x": "Coin", "y": "Beta"}, color=betas,
                     color_continuous_scale="RdYlGn")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Amplification vs Volume
        fig = px.scatter(
            x=[s.volume_surge for s in alt_signals[:25]],
            y=[s.amplification_ratio for s in alt_signals[:25]],
            text=[s.symbol.replace("/USDT", "") for s in alt_signals[:25]],
            size=[max(s.score, 10) for s in alt_signals[:25]],
            title="Amplification Ratio vs Volume Surge",
            labels={"x": "Volume Surge (x avg)", "y": "Amplification Ratio"},
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("### Scanner Summary")
    s1, s2, s3, s4 = st.columns(4)
    amplifying = [s for s in alt_signals if s.is_amplifying]
    s1.metric("Alts Scanned", len(alt_signals))
    s2.metric("Actively Amplifying", len(amplifying))
    s3.metric("Avg Beta (Top 10)", f"{np.mean([s.beta_vs_btc for s in alt_signals[:10]]):.1f}x")
    s4.metric("Avg Score (Top 10)", f"{np.mean([s.score for s in alt_signals[:10]]):.0f}")

else:
    st.info("No alt signals available. This could be due to rate limiting — try again in a minute.")

st.markdown("---")

# Row 3: Trading State
st.subheader("📊 Trading Status")
state = get_trading_state()

if state:
    positions = state.get("positions", {})
    if positions:
        st.success(f"🟢 {len(positions)} open positions")
        pos_data = []
        for sym, p in positions.items():
            pos_data.append({
                "Symbol": sym,
                "Entry Price": f"${p['entry_price']:.4f}",
                "Entry Time": p["entry_time"],
                "BTC Momentum at Entry": f"{p['entry_btc_momentum']:.0f}",
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No open positions")
    st.metric("Total Trades", state.get("trade_count", 0))
else:
    st.info("Trading engine not running. Start with: `python3 live_trader.py`")


# Auto refresh
if auto_refresh:
    time.sleep(300)
    st.rerun()
