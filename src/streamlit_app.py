"""Streamlit dashboard for the Futures Execution Explorer.

Run:  streamlit run src/streamlit_app.py   ->   http://localhost:8501

A polished, responsive replacement for the stdlib-HTTP `app.py`. The backend pure
functions are reused verbatim; this module only adds the page, chained caching, and
Plotly wiring. Caching is keyed on primitive args so a knob only invalidates what
depends on it — nudging `fill_target` re-runs *only* the backtest, not ranges/regime/ePDF.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import viz_plotly as viz  # noqa: E402

# Loader/scan helpers reused verbatim from the HTTP app (matches scripts/run_v1.py).
from app import _load_source, _scan_contracts, _scan_markets  # noqa: E402
from epdf import build_epdf  # noqa: E402
from order_mgmt.agent.benchmarks import (  # noqa: E402
    evaluate_agent_benchmarks,
    evaluate_tail_strategies,
)
from order_mgmt.agent.loader import (  # noqa: E402
    find_agent_csv,
    load_agent_series,
    load_market_for_agent,
)
from order_mgmt.agent.metrics import evaluate_agent_execution  # noqa: E402
from order_mgmt.backtest import run_backtest_rolling  # noqa: E402
from order_mgmt.baselines import vwap_baseline  # noqa: E402
from order_mgmt.ticks import resolve_tick  # noqa: E402
from plot_volume import _compute_stats  # noqa: E402
from plotting import build_histogram_figure  # noqa: E402
from ranges import compute_all_ranges  # noqa: E402
from regime import compute_ewma_series  # noqa: E402

TAU_VALUES = [1, 5, 10, 15, 30, 60]
AGENT_CAPS = (4, 6, 8, 10, 12, 16)  # chase-cap sweep (ticks adverse before stop-out)

_AGENT_EXPLAINER = """
**What "agent value" measures.** The AI agent decides *what* and *when* to trade; our module
decides *how* to fill each parent order. Value-add is the agent's realised execution price
versus a naïve benchmark, in ticks. We benchmark against the **OHLC window open**
(market-on-decision), not the agent's own CSV price, because the agent price can sit on a
different contract than the rolled OHLC series — that basis would otherwise masquerade as
slippage. Positive ticks ⇒ the execution layer *beat* trading at the open.

**The headline.** Posting a passive regime limit `ℓ*` (the largest offset whose per-regime
ePDF survival ≥ the fill-rate target) earns a solid **median** improvement, but the
*mean* is dragged to ≈ 0 by a heavy tail: when price runs away, the unfilled order chases
and pays. **Fill-rate is the robust quantity; the mean lives or dies on the tail.**

**The winner — regime-limit + chase-cap.** Keep the limit's upside but stop out at a fixed
`cap` ticks of adverse move (the *chase-cap* sweep above). The asymmetry — uncapped upside,
bounded downside — turns the mean **positive** while keeping the median, and shrinks the
5th-percentile tail dramatically. The best cap is market-dependent (dial it above).

**Strategy levers (Stream D)** you can compose on top of the picker:
- **Cost-aware ℓ\\*** (`pick_ell_star_cost_aware`): instead of a fill-rate target, maximise
  `p(ℓ)·ℓ − (1−p(ℓ))·chase_cost`. It never picks worse than market-on-open and pulls the
  limit in as the chase cost rises.
- **Chase-at-mid** (`chase_price(policy="mid")`): on an unfilled order, execute at the
  window mid `(H+L)/2` instead of the close — strictly better mean *and* tail.
- **Early-chase** (`simulate_early_chase`): bail as soon as price moves a trigger distance
  against the limit rather than waiting for the deadline — the intrabar cousin of the cap.

**Caveat (genericity, not over-fitting).** A regime-conditioning ablation found the 27-cell
edge over a single pooled ePDF is only ~0.02–0.05 ticks of mean — the chase-cap, not the
conditioning, is doing the heavy lifting. The split here is a single time-ordered hold-out
(regime fit on pre-trade history, evaluated on the trade window); a full in/out-of-sample
parameter leaderboard is the natural next step.
"""

st.set_page_config(page_title="Futures Execution Explorer", layout="wide")

# Hide Streamlit's top-right "running man" status widget (we show our own spinner).
st.markdown(
    "<style>[data-testid='stStatusWidget']{display:none;}</style>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data layer — chained so each cache only busts on the args it depends on.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_source(source: str):
    """(df_ohlcv, ticker, first_stem) or None — roll-aware market or single contract."""
    return _load_source(source)


@st.cache_data(show_spinner=False)
def compute_stats(source: str):
    """(daily, tick_inferred, tick_eff, proper_days, n_green, n_total).

    Ranges use the *inferred* tick and the backtest uses the spec-resolved tick,
    mirroring the existing UI / scripts/run_v1.py exactly.
    """
    df, _ticker, first_stem = load_source(source)
    daily, tick, proper_days, n_green, n_total = _compute_stats(df)
    tick_eff = resolve_tick(first_stem, tick)
    return daily, tick, tick_eff, proper_days, n_green, n_total


@st.cache_data(show_spinner=False)
def compute_ranges(source: str, tau: int):
    df, _ticker, _stem = load_source(source)
    _daily, tick, _tick_eff, proper_days, _g, _t = compute_stats(source)
    return compute_all_ranges(df, tau, tick, proper_days)


@st.cache_data(show_spinner=False)
def compute_ewma(source: str, tau: int, half_life: int):
    t_list, ell_r, _ell_u, _ell_d, vol_list, _dx = compute_ranges(source, tau)
    return compute_ewma_series(t_list, ell_r, vol_list, half_life)


@st.cache_data(show_spinner=False)
def compute_epdf(source: str, tau: int, half_life: int, M: int, N: int, K: int, j_start: int):
    t_list, _ell_r, ell_u, ell_d, _vol, dx_list = compute_ranges(source, tau)
    ewma_range, ewma_vol = compute_ewma(source, tau, half_life)
    counts_RU, counts_RD, _thr = build_epdf(
        t_list, ell_u, ell_d, list(ewma_vol), list(ewma_range), dx_list,
        M=M, N=N, K=K, j_start=j_start,
    )
    return counts_RU, counts_RD


@st.cache_data(show_spinner=False)
def run_bt(source: str, tau: int, half_life: int, M: int, N: int, K: int,
           j_start: int, fill_target: float, side: str):
    df, _ticker, _stem = load_source(source)
    _daily, _tick, tick_eff, proper_days, _g, _t = compute_stats(source)
    return run_backtest_rolling(
        df, tau=tau, tick=tick_eff, proper_days=proper_days, side=side,
        fill_rate_target=fill_target, half_life=half_life, M=M, N=N, K=K, j_start=j_start,
    )


@st.cache_data(show_spinner=False)
def run_vwap(source: str, tau: int, j_start: int, side: str):
    df, _ticker, _stem = load_source(source)
    t_list = compute_ranges(source, tau)[0]
    _daily, _tick, tick_eff, _pd, _g, _t = compute_stats(source)
    return vwap_baseline(df, t_list[j_start:], tau=tau, tick=tick_eff, side=side)


@st.cache_data(show_spinner=False)
def run_sweep(source: str, tau: int, half_life: int, M: int, N: int, K: int,
              j_start: int, fill_grid: tuple[float, ...], side: str):
    """One backtest per grid point (each cached individually via run_bt)."""
    out = []
    for ft in fill_grid:
        bt = run_bt(source, tau, half_life, M, N, K, j_start, ft, side)
        out.append((ft, bt.fill_rate, bt.avg_slippage_ticks))
    return out


# --- Agent execution-value layer (AIAgent_*.csv vs OHLC) -------------------
@st.cache_data(show_spinner=False)
def load_agent_market(market_name: str):
    """(agent_series, df_ohlcv, tick, proper_days) for a market shipping an AIAgent CSV.

    Roll-aware OHLC + spec tick via load_market_for_agent — identical wiring to
    scripts/run_v1.py. Returns None if no AIAgent file or no usable OHLC.
    """
    market_dir = DATA / market_name
    csv = find_agent_csv(market_dir)
    if csv is None:
        return None
    agent = load_agent_series(csv, market_name)
    df_ohlcv, tick, proper_days, _stem = load_market_for_agent(market_dir)
    if df_ohlcv.empty:
        return None
    return agent, df_ohlcv, tick, proper_days


@st.cache_data(show_spinner=False)
def run_agent_eval(market_name: str, tau: int, half_life: int, M: int, N: int, K: int,
                   j_start: int, fill_target: float):
    loaded = load_agent_market(market_name)
    if loaded is None:
        return None
    agent, df_ohlcv, tick, proper_days = loaded
    return evaluate_agent_execution(
        agent, df_ohlcv, tick, proper_days, tau=tau, half_life=half_life,
        M=M, N=N, K=K, fill_rate_target=fill_target, j_start=j_start,
    )


@st.cache_data(show_spinner=False)
def run_agent_benchmarks(market_name: str, tau: int, half_life: int, M: int, N: int, K: int,
                         j_start: int, fill_target: float):
    loaded = load_agent_market(market_name)
    if loaded is None:
        return None
    agent, df_ohlcv, tick, proper_days = loaded
    return evaluate_agent_benchmarks(
        agent, df_ohlcv, tick, proper_days, tau=tau, half_life=half_life,
        M=M, N=N, K=K, fill_rate_target=fill_target, j_start=j_start, seed=0,
    )


@st.cache_data(show_spinner=False)
def run_agent_caps(market_name: str, tau: int, half_life: int, M: int, N: int, K: int,
                   j_start: int, fill_target: float):
    loaded = load_agent_market(market_name)
    if loaded is None:
        return None
    agent, df_ohlcv, tick, proper_days = loaded
    return evaluate_tail_strategies(
        agent, df_ohlcv, tick, proper_days, tau=tau, half_life=half_life,
        M=M, N=N, K=K, fill_rate_target=fill_target, j_start=j_start, caps=AGENT_CAPS,
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
markets = _scan_markets()
contracts = _scan_contracts()
agent_markets = {p.name: p for p in markets if find_agent_csv(p) is not None}

source_options = (
    [f"m:{p.name}" for p in markets]
    + [f"c:{p.stem}" for p in contracts]
)
source_labels = {
    **{f"m:{p.name}": f"{p.name}  (roll-aware)" for p in markets},
    **{f"c:{p.stem}": f"{p.stem}  ({p.parent.name})" for p in contracts},
}

with st.sidebar:
    st.header("Controls")

    st.subheader("Data")
    if not source_options:
        st.error(f"No data found under {DATA}")
        st.stop()
    source = st.selectbox("Market / contract", source_options,
                          format_func=lambda s: source_labels.get(s, s))

    st.subheader("Regime model")
    tau = st.select_slider("τ — holding period (min)", options=TAU_VALUES, value=5)
    half_life = st.slider("half-life (windows)", 5, 200, 20)
    n_range = st.slider("N — range states", 2, 6, 3)
    n_vol = st.slider("M — volume states", 2, 6, 3)
    k_dx = st.slider("K — Δx direction states", 2, 6, 3)
    j_start = st.slider("j_start — warm-up windows", 50, 500, 200, step=10,
                        help="Windows skipped before the ePDF / backtest start accumulating.")

    st.subheader("Strategy")
    fill_target = st.slider("fill-rate target", 0.30, 0.90, 0.60, step=0.05)
    side_choice = st.radio("Side", ["Buy", "Sell", "Both"], index=2, horizontal=True)
    auto_bt = st.checkbox("Run backtest", value=True,
                          help="Uncheck for faster regime tuning (skips the heavy backtest).")
    agent_on = st.checkbox("Agent execution-value", value=False,
                           help="Score the AI agent's fills vs benchmarks on its AIAgent_*.csv "
                                "(heavier; uses the regime/strategy knobs above).")

    with st.expander("Fill-target sweep (Pareto)"):
        st.caption("Runs one backtest per grid point per side — heavier.")
        sweep_on = st.checkbox("Enable sweep", value=False)
        sw_lo, sw_hi = st.slider("target range", 0.30, 0.90, (0.40, 0.80), step=0.05)
        sw_step = st.select_slider("step", options=[0.05, 0.10, 0.20], value=0.10)

sides = {"Buy": ["buy"], "Sell": ["sell"], "Both": ["buy", "sell"]}[side_choice]

# ---------------------------------------------------------------------------
# Header + load
# ---------------------------------------------------------------------------
st.title("Futures Execution Explorer")
st.caption(
    "Regime-conditioned limit-order execution — slice parent orders by mining "
    "volatility/volume regimes from 1-min OHLC futures data. Backtest is v2 (no-lookahead)."
)

loaded = load_source(source)
if loaded is None:
    st.warning("Source not found or no usable data.")
    st.stop()
_df, ticker, _first_stem = loaded

with st.spinner("Computing ranges, EWMA and ePDFs…"):
    daily, tick, tick_eff, proper_days, n_green, n_total = compute_stats(source)
    t_list, ell_r, ell_u, ell_d, vol_list, dx_list = compute_ranges(source, tau)
    ewma_range, ewma_vol = compute_ewma(source, tau, half_life)
    counts_RU, counts_RD = compute_epdf(source, tau, half_life, n_vol, n_range, k_dx, j_start)

# Backtest + VWAP (only if requested).
bt_results: dict[str, object] = {}
vwap_results: dict[str, object] = {}
if auto_bt:
    with st.spinner("Running no-lookahead backtest…"):
        for s in sides:
            bt_results[s] = run_bt(source, tau, half_life, n_vol, n_range, k_dx,
                                   j_start, fill_target, s)
            vwap_results[s] = run_vwap(source, tau, j_start, s)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
base_kpis = 3
kpi_cols = st.columns(base_kpis + (2 * len(sides) if auto_bt else 0))
pct = 100 * n_green / n_total if n_total else 0.0
kpi_cols[0].metric("Active days", f"{n_green}/{n_total}", f"{pct:.0f}%")
kpi_cols[1].metric("Tick size", f"{tick_eff:g}")
kpi_cols[2].metric("Valid windows", f"{len(ell_r):,}", f"τ = {tau} min")

if auto_bt:
    for i, s in enumerate(sides):
        bt = bt_results[s]
        vw = vwap_results[s]
        vw_avg = float(np.mean(vw.slippage_ticks)) if vw.slippage_ticks else 0.0
        c_fill = kpi_cols[base_kpis + 2 * i]
        c_slip = kpi_cols[base_kpis + 2 * i + 1]
        c_fill.metric(f"{s.capitalize()} fill-rate", f"{bt.fill_rate:.1%}")
        c_slip.metric(
            f"{s.capitalize()} avg slip", f"{bt.avg_slippage_ticks:+.2f}t",
            f"{bt.avg_slippage_ticks - vw_avg:+.2f}t vs VWAP",
        )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dq, tab_rng, tab_reg, tab_epdf, tab_bt, tab_sweep, tab_agent = st.tabs(
    ["Data Quality", "Ranges", "Regimes", "ePDFs", "Backtest", "Sweep", "Agent value"]
)

with tab_dq:
    st.caption("Green spans = active days (traded minutes ≥ 90 % of the dataset max).")
    vol_daily = _df["volume"].resample("D").sum()
    vol_daily = vol_daily[vol_daily > 0]
    st.plotly_chart(viz.volume_fig(daily, vol_daily, ticker), use_container_width=True)

with tab_rng:
    st.caption(
        f"R = high - low  |  R_U = high - open  |  R_D = open - low  "
        f"(ticks, τ = {tau} min). Identity: R ≈ R_U + R_D."
    )
    fig_hist = build_histogram_figure(ell_r, ell_u, ell_d, tau, tick, ticker)
    st.pyplot(fig_hist)

with tab_reg:
    st.caption(
        f"EWMA Range / Volume (half-life = {half_life} windows) and "
        f"Δx = open[t+τ] - open[t] (ticks). Colour bands = quantile-based states."
    )
    st.plotly_chart(
        viz.regime_panel_fig(t_list, ewma_range, n_range,
                             f"EWMA Range  ({n_range} states)"),
        use_container_width=True,
    )
    st.plotly_chart(
        viz.regime_panel_fig(t_list, ewma_vol, n_vol,
                             f"EWMA Volume  ({n_vol} states)"),
        use_container_width=True,
    )
    st.plotly_chart(
        viz.regime_panel_fig(t_list, np.asarray(dx_list, dtype=float), k_dx,
                             f"Δx  ({k_dx} states)"),
        use_container_width=True,
    )

with tab_epdf:
    st.caption(
        "Per-regime cell (m = volume, n = range, k = direction): mean R_U / R_D in "
        f"ticks, hover for window counts. Skips j < {j_start} (warm-up)."
    )
    st.plotly_chart(
        viz.epdf_heatmap_fig(counts_RU, counts_RD, n_vol, n_range, k_dx),
        use_container_width=True,
    )
    with st.expander("Raw regime table"):
        rows = []
        for m in range(1, n_vol + 1):
            for n_st in range(1, n_range + 1):
                for k in range(1, k_dx + 1):
                    ru = counts_RU.get((m, n_st, k))
                    rd = counts_RD.get((m, n_st, k))
                    if not ru:
                        continue
                    n_obs = sum(ru.values())
                    mean_ru = sum(e * c for e, c in ru.items()) / n_obs
                    mean_rd = (sum(e * c for e, c in rd.items()) / sum(rd.values())
                               if rd else 0.0)
                    rows.append({"m": m, "n": n_st, "k": k, "n_obs": n_obs,
                                 "mean_R_U": round(mean_ru, 2), "mean_R_D": round(mean_rd, 2)})
        if rows:
            df_tab = pd.DataFrame(rows)
            st.dataframe(
                df_tab.style.background_gradient(subset=["mean_R_U", "mean_R_D"], cmap="viridis"),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No populated regime cells — lower j_start or use fewer states.")

with tab_bt:
    if not auto_bt:
        st.info("Backtest is off — enable **Run backtest** in the sidebar.")
    else:
        st.caption(
            f"Strategy v2 (no-lookahead) vs VWAP. TWAP = open is the zero baseline. "
            f"fill-rate target = {fill_target:.2f}."
        )
        for s in sides:
            bt = bt_results[s]
            vw = vwap_results[s]
            st.markdown(
                f"**{s.capitalize()}** — n={bt.n_decisions:,}, fill={bt.fill_rate:.1%}, "
                f"avg={bt.avg_slippage_ticks:+.2f}t, median={bt.median_slippage_ticks:+.2f}t"
            )
            st.plotly_chart(
                viz.backtest_hist_fig(bt.slippage_ticks, vw.slippage_ticks, s, ticker),
                use_container_width=True,
            )

with tab_sweep:
    if not sweep_on:
        st.info("Enable **Fill-target sweep** in the sidebar to compute the Pareto curve.")
    else:
        grid = tuple(round(x, 2) for x in np.arange(sw_lo, sw_hi + 1e-9, sw_step))
        st.caption(f"Sweeping fill-rate target over {list(grid)} - {len(grid)} points, {len(sides)} side(s).")
        prog = st.progress(0.0)
        total = len(sides)
        for i, s in enumerate(sides):
            with st.spinner(f"Sweeping {s}…"):
                pts = run_sweep(source, tau, half_life, n_vol, n_range, k_dx, j_start, grid, s)
            st.plotly_chart(viz.pareto_fig(pts, s, current_target=fill_target),
                            use_container_width=True)
            prog.progress((i + 1) / total)
        prog.empty()

with tab_agent:
    st.caption(
        "Does regime-conditioned execution add value to the **AI agent's** trading? The "
        "agent's parent orders are the rows of its AIAgent_*.csv where its signed position "
        "changes. We fill each on the OHLC data using the regime ℓ* picker, and benchmark "
        "against the **OHLC window open** (basis-immune — positive ticks beat market-on-decision)."
    )
    if not agent_markets:
        st.info("No AIAgent_*.csv found under the data folder.")
    elif not agent_on:
        st.info("Enable **Agent execution-value** in the sidebar to run this analysis "
                "(it reuses the τ / half-life / M / N / K / fill-rate knobs).")
    else:
        names = list(agent_markets)
        default_idx = names.index(source[2:]) if source.startswith("m:") and source[2:] in agent_markets else 0
        a_market = st.selectbox("Agent market", names, index=default_idx)

        with st.spinner(f"Evaluating {a_market} agent execution (regime fit on pre-trade history)…"):
            res = run_agent_eval(a_market, tau, half_life, n_vol, n_range, k_dx, j_start, fill_target)
            bmk = run_agent_benchmarks(a_market, tau, half_life, n_vol, n_range, k_dx, j_start, fill_target)
            caps = run_agent_caps(a_market, tau, half_life, n_vol, n_range, k_dx, j_start, fill_target)

        if res is None or res.n_decisions == 0:
            st.warning("No fillable agent decisions for this market/regime — try a lower j_start.")
        else:
            a = st.columns(6)
            a[0].metric("Decisions", f"{res.n_decisions:,}")
            a[1].metric("Fill-rate", f"{res.fill_rate:.1%}")
            a[2].metric("Mean shortfall", f"{res.mean_shortfall_ticks:+.2f}t")
            a[3].metric("Median shortfall", f"{res.median_shortfall_ticks:+.2f}t")
            a[4].metric("Value vs market", f"{res.value_add_vs_market_ticks:+.2f}t")
            a[5].metric("Value vs VWAP", f"{res.value_add_vs_vwap_ticks:+.2f}t")

            if bmk is not None:
                st.markdown("**Benchmark ladder** — static ℓ* schemes on identical orders.")
                st.plotly_chart(viz.agent_benchmark_fig(bmk, a_market), use_container_width=True)

            if caps is not None and any(caps[f"cap{c}"]["shortfall"].size for c in AGENT_CAPS):
                cap_means = {
                    c: (float(np.mean(caps[f"cap{c}"]["shortfall"]))
                        if caps[f"cap{c}"]["shortfall"].size else -np.inf)
                    for c in AGENT_CAPS
                }
                best_cap = max(cap_means, key=cap_means.get)
                st.markdown(
                    f"**Chase-cap frontier** — the winning *regime-limit + chase-cap* policy. "
                    f"Best cap by mean here: **{best_cap} ticks**."
                )
                st.plotly_chart(viz.agent_cap_sweep_fig(caps, AGENT_CAPS, a_market),
                                use_container_width=True)
                st.plotly_chart(
                    viz.agent_shortfall_fig(
                        caps[f"cap{best_cap}"]["shortfall"], caps["regime"]["shortfall"],
                        f"regime-limit + cap{best_cap}", "regime-limit (uncapped)", a_market,
                    ),
                    use_container_width=True,
                )

            with st.expander("What this means — agentic value & the strategy levers", expanded=True):
                st.markdown(_AGENT_EXPLAINER)
