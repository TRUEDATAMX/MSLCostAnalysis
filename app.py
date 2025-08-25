import streamlit as st
import numpy as np
import pandas as pd
from typing import Literal, Tuple
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import glob, os

# ============================================================
# Helper functions (from your notebook)
# ============================================================
_Method = Literal["normal", "wilson", "exact", "agresti_coull", "bootstrap"]

def _ci_for_p_at_n(p: float, n: int, confidence: float, method: _Method, n_bootstrap: int = 2000) -> Tuple[float, float]:
    """Compute a CI for a binomial proportion assuming expected count x = round(p*n).
    Used for planning (before collecting data)."""
    if n <= 0:
        raise ValueError("n must be positive")
    x = int(round(p * n))
    alpha = 1 - confidence

    if method == "bootstrap":
        data = np.array([1] * x + [0] * (n - x))
        boots = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            boots[i] = np.mean(np.random.choice(data, size=n, replace=True))
        lower, upper = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lower), float(upper)

    sm_method = {"exact": "beta"}.get(method, method)
    lower, upper = proportion_confint(count=x, nobs=n, alpha=alpha, method=sm_method)
    return float(lower), float(upper)

def _ci_half_width(p: float, n: int, confidence: float, method: _Method, n_bootstrap: int = 2000) -> float:
    lo, hi = _ci_for_p_at_n(p, n, confidence, method, n_bootstrap=n_bootstrap)
    return (hi - lo) / 2.0

def required_sample_size_ci(
    p: float,
    E: float,
    confidence: float = 0.95,
    method: _Method = "wilson",
    n_bootstrap: int = 2000,
    max_iter: int = 2_000_000,
) -> int:
    """Invert the CI: minimal n such that CI half-width <= E for given confidence."""
    if not (0 < p < 1):
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
    if E <= 0:
        raise ValueError("E must be > 0")

    Z = norm.ppf(1 - (1 - confidence) / 2)
    n0 = int(np.ceil((Z**2) * p * (1 - p) / (E**2)))
    n0 = max(30, n0)

    n_low = max(30, int(n0 * 0.5))
    n_high = max(n_low + 1, int(n0 * 1.5))
    hw_high = _ci_half_width(p, n_high, confidence, method, n_bootstrap=n_bootstrap)

    while hw_high > E:
        n_low = n_high
        n_high = int(n_high * 2)
        if n_high > max_iter:
            raise RuntimeError("required_sample_size_ci: exceeded max_iter")
        hw_high = _ci_half_width(p, n_high, confidence, method, n_bootstrap=n_bootstrap)

    lo, hi = n_low + 1, n_high
    ans = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        hw = _ci_half_width(p, mid, confidence, method, n_bootstrap=n_bootstrap)
        if hw <= E:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def achieved_confidence_from_n(
    n: int,
    p: float,
    E: float,
    method: _Method = "wilson",
    n_bootstrap: int = 2000,
    tol: float = 1e-4,
) -> float:
    """Given n, p, E, find max confidence with CI half-width <= E (binary search)."""
    if n <= 0:
        raise ValueError("n must be positive")
    lo, hi = 0.50, 0.999999
    best = lo
    for _ in range(60):
        mid = (lo + hi) / 2
        hw = _ci_half_width(p, n, mid, method, n_bootstrap=n_bootstrap)
        if hw <= E:
            best = mid
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return best

def apply_fpc(n: int, population_size: int) -> int:
    """Finite Population Correction for proportions: n_adj = n / (1 + (n - 1)/N)"""
    if population_size <= 0:
        return n
    n_adj = n / (1 + (n - 1) / population_size)
    return int(np.ceil(n_adj))

def cost_per_call_range(transcription_min=0.0005, transcription_max=0.0014, ai_unit=0.001):
    """Return (min_cost, max_cost) per call."""
    return (transcription_min + ai_unit, transcription_max + ai_unit)

def total_cost(n: int, per_call_cost: float) -> float:
    return n * per_call_cost

# ============================================================
# Streamlit App
# ============================================================
st.title("📊 Statistical Justification for Call Sampling")

# ---------------------------
# Objectives (verbatim)
# ---------------------------
st.header("Objectives")
st.markdown("""
The operational platform developed for Marketing Storm Leads must achieve the following goals:

* Sales/Appointment Calls: Every call that leads to a sales transfer/appointment will be fully processed and analyzed. These are high-value events, so full coverage is justified.

* General QA Calls: Calls unrelated to direct sales will be analyzed only in a sampled subset to save costs. Here, statistics will help us decide how many calls we need to monitor without overspending.

The proposal already covers the first point, here we discuss why analyzing only a small subset of calls would suffice the second one. 
""")

# ---------------------------
# Background (verbatim)
# ---------------------------
st.header("Background")
st.markdown("""
Our system is designed to keep costs as low as possible while still meeting business needs. We run transcription using Whisper.cpp, a highly efficient open-source model, deployed on spot cloud instances, the cheapest type of virtual machines that can handle our demand.
This setup ensures that we are already operating at the lowest feasible infrastructure cost.
""")

# ---------------------------
# Problem Statement (verbatim)
# ---------------------------
st.header("Problem Statement")
st.markdown("""
The following data is assumed to be true:
* MSL generates a total of 30M monthly calls
* The price of processing a single call, in average, is 0.0024 USD

The problem to be solved is to ensure calls' quality while minimizing costs.
Processing the totality of monthly calls would yield a total cost of:
""", unsafe_allow_html=True)

# Preserve your HTML formatting line
st.markdown("""&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 30M calls * 0.0024 USD/call =  72175,4064 USD""", unsafe_allow_html=True)

st.markdown("""
Which makes analyzing the totality of calls unfeasible, subject to the assumptions presented above.
<br/><br/>

To solve this, we propose a sampling strategy to get a representative result for Call Quality from a small subset of the entire call population.
In order to achieve this, we consider the following key statistical questions:

1 - How many calls do we need to sample to clearly understand the true error/bad-quality proportion $p$ within margin $E$ at confidence level $C$?

2 - What confidence do we gain if we sample $N$ calls instead of all calls?

Statistically, this is a binomial proportion estimation problem. While normal approximations are common, we’ll also look at exact (Clopper–Pearson) intervals for small or skewed $p$, which is a safe assumption, since we don't expect to have a big proportion of bad quality calls.
""", unsafe_allow_html=True)

# ---------------------------
# Statistical Background (verbatim)
# ---------------------------
st.header("Statistical Background")
st.markdown("""
Two approaches are useful for our purposes:

1 - Sampling to detect proportion of "bad calls" (e.g., abuse, non-compliance).

2 - Confidence intervals: Estimating error bounds on proportions.

Formula for required sample size:
""")
st.latex(r"n=\frac{Z^{2} \cdot p(1-p)}{E^2}")

st.markdown("""
where $p$ = assumed proportion, $E$ = margin of error, $Z$ = confidence level critical value.
For small $p$, the standard is use Clopper-Pearson intervals (binomial exact).
""")

# ---------------------------
# Business Intuition (verbatim)
# ---------------------------
st.header("Business Intuition")
st.markdown("""
The intuition behind these approaches is that they help us to answer the following business questions:

* How many calls do we need to analyze to be confident in our quality estimates?

Instead of blindly reviewing all calls (which is expensive), we want to know the minimum representative sample size that allows us to detect problems in quality with a high degree of confidence.

* What is the trade-off between cost and confidence?

Each additional call analyzed has an associated cost (transcription + AI analysis). By quantifying how confidence grows with sample size, we can decide when further investment no longer provides meaningful business benefit.

* What level of risk is acceptable?
By formalizing the sampling design, we can translate business risk (e.g., missing abusive calls, or failing to detect a drop in quality) into statistical terms (e.g., confidence intervals, detection thresholds).

Together, these statistical methods give us a rigorous, cost-efficient strategy for deciding how many calls to analyze, while ensuring that our analysis supports the business strategy without overspending.
""")

# ---------------------------
# Analysis (verbatim intro text)
# ---------------------------
st.header("Analysis")
st.markdown("""
This section provides interactive tools to explore sample-size, confidence, and cost trade-offs for the QA sampling strategy.  
Use the widgets to vary assumptions and immediately see their impact.

Notes  
- We model QA events (e.g., "bad-quality" calls) as a Bernoulli process; confidence intervals come from binomial methods (e.g., Wilson, Exact/Clopper–Pearson).  
- We include finite population correction (FPC) if the QA sample is not tiny relative to the eligible QA pool.  
""")

# ---------------------------
# Interactive: Required Sample Size by Method
# ---------------------------
st.subheader("Required Sample Size by Method")
col1, col2, col3 = st.columns(3)
with col1:
    p = st.slider("p (bad rate)", 0.01, 0.99, 0.5, 0.01)
with col2:
    E = st.slider("±E (margin of error)", 0.001, 0.1, 0.02, 0.001, format="%.3f")
with col3:
    confidence = st.slider("Confidence", 0.80, 0.999, 0.95, 0.001, format="%.3f")

colA, colB = st.columns(2)
with colA:
    Nqa = st.number_input("QA Pool N", value=30_000_000, step=1_000)
with colB:
    use_fpc = st.checkbox("Apply FPC", value=True)

methods = st.multiselect("Methods", ['wilson', 'exact', 'agresti_coull', 'normal'],
                         default=['wilson','exact'])

rows = []
for m in methods:
    n_req = required_sample_size_ci(p=p, E=E, confidence=confidence, method=m)
    n_adj = apply_fpc(n_req, Nqa) if use_fpc else n_req
    rows.append((m, n_req, n_adj))
st.dataframe(pd.DataFrame(rows, columns=['Method', 'Required n', 'Required n (FPC)']), use_container_width=True)

# ---------------------------
# Achieved Confidence vs Sample Size
# ---------------------------
st.subheader("Achieved Confidence vs Sample Size")
ns = list(range(200, 15001, 200))
conf_vals = [achieved_confidence_from_n(n, p, E, method='wilson') for n in ns]
fig, ax = plt.subplots()
ax.plot(ns, conf_vals)
ax.set_xlabel("Sample size (n)")
ax.set_ylabel("Achieved confidence")
ax.set_title(f"Confidence vs Sample Size (method=wilson, p={p}, E=±{E})")
ax.grid(True)
st.pyplot(fig)

# ---------------------------
# Cost vs Sample Size & Diminishing Returns
# ---------------------------
st.subheader("Cost vs Sample Size (per-call range)")
ns2 = np.arange(200, 20001, 200)
cmin, cmax = cost_per_call_range(0.0005, 0.0014, 0.001)
total_min = ns2 * cmin
total_max = ns2 * cmax
fig2, ax2 = plt.subplots()
ax2.plot(ns2, total_min, label="Lower bound cost")
ax2.plot(ns2, total_max, label="Upper bound cost")
ax2.set_xlabel("Sample size (n)")
ax2.set_ylabel("Total variable cost (USD)")
ax2.set_title("Cost vs Sample Size")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ---------------------------
# Cost–Confidence Trade-off Explorer (generic)
# ---------------------------
st.subheader("Cost–Confidence Trade-off Explorer")
method_cc = st.selectbox("CI Method", ['wilson', 'exact', 'agresti_coull', 'normal'], index=0, key="method_cc")
n_req_cc = required_sample_size_ci(p=p, E=E, confidence=confidence, method=method_cc)
n_chosen_cc = apply_fpc(n_req_cc, Nqa) if use_fpc else n_req_cc
cc_min, cc_max = cmin, cmax
st.write(f"Required n (no FPC): {n_req_cc:,}")
st.write(f"Required n (with FPC): {n_chosen_cc:,}")
st.write(f"Per-call cost range: ${cc_min:.6f} — ${cc_max:.6f}")
st.write(f"Total cost range for chosen n: ${total_cost(n_chosen_cc, cc_min):,.2f} — ${total_cost(n_chosen_cc, cc_max):,.2f}")

# ============================================================
# Exercise with actual MSL monthly data (your notebook content)
# ============================================================
st.header("Exercise with actual MSL monthly data")

# Read & merge CSVs under ./data
@st.cache_data(show_spinner=True)
def load_call_data(data_dir="data"):
    files = glob.glob(os.path.join(data_dir, "fastcallcenter*.csv"))
    if not files:
        return None, []
    df_list = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df, files

df_raw, files_found = load_call_data("data")

if not files_found:
    st.warning("No CSVs found in ./data/. Place your files there (e.g., fastcallcenter_call_log_report_*.csv).")
else:
    # Preserve your notebook steps
    df = df_raw.copy()
    total_before_filters = len(df)

    # Recording Length (Seconds) numeric
    if "Recording Length (Seconds)" in df.columns:
        df["Recording Length (Seconds)"] = pd.to_numeric(df["Recording Length (Seconds)"], errors="coerce")
        df = df[df["Recording Length (Seconds)"] >= 30]
    else:
        # if not present, try to coerce "Recording Length" as seconds
        if "Recording Length" in df.columns:
            df["Recording Length"] = pd.to_numeric(df["Recording Length"], errors="coerce")
            df = df[df["Recording Length"] >= 30]

    # Identify "Transfer" (appointments) vs QA pool
    df["Log Type"] = df["Log Type"].astype(str)
    transferred_calls = df[df["Log Type"].str.lower() == "transfer"]
    qa_calls = df[df["Log Type"].str.lower() != "transfer"]

    total_calls = len(df)
    transferred_count = len(transferred_calls)
    qa_count = len(qa_calls)

    st.write("### Call Population Summary")
    st.write(f"Total calls before filtering: **{total_before_filters:,}**")
    st.write(f"Total calls after filtering: **{total_calls:,}**")
    st.write(f"Transferred (qualification, 100% processed): **{transferred_count:,}**")
    st.write(f"Remaining QA calls (subject to sampling): **{qa_count:,}**")

    # Mandatory qualification cost (your constants)
    transcription_min = 0.0005
    transcription_max = 0.0014
    ai_cost = 0.001

    qualification_cost_min = transferred_count * (transcription_min + ai_cost)
    qualification_cost_max = transferred_count * (transcription_max + ai_cost)

    st.write("### Mandatory Qualification Cost")
    st.write(f"Low estimate: **${qualification_cost_min:,.2f}**")
    st.write(f"High estimate: **${qualification_cost_max:,.2f}**")

    # ---------------------------
    # 🔶 Simulator with actual MSL data (interactive)
    # ---------------------------
    st.subheader("Simulator: QA Sampling on Actual Monthly Data")

    colL, colM, colR = st.columns(3)
    with colL:
        p_qa = st.slider("Assumed bad-call rate (p)", 0.01, 0.99, 0.50, 0.01)
        st.caption("📊 *The estimated proportion of problematic calls.*\n\n"
                "Example: `0.10` → assume 10% of calls are low-quality.")
    with colM:
        E_qa = st.slider("Margin of error (±E)", 0.001, 0.10, 0.02, 0.001, format="%.3f")
        st.caption("🎯 *How precise you want your estimate to be.*\n\n"
                "Example: `0.02` → your estimate of bad-call rate is within ±2%.")
    with colR:
        conf_qa = st.slider("Confidence level", 0.80, 0.999, 0.95, 0.001, format="%.3f")
        st.caption("🔒 *How confident you want to be in your results.*\n\n"
                "Example: `0.95` → 95% of the time, the interval will contain the true rate.")

    colM1, colM2, colM3 = st.columns(3)
    with colM1:
        method_qa = st.selectbox("CI Method", ['wilson', 'exact', 'agresti_coull', 'normal'], index=0, key="method_qa")
        st.caption("📐 *Statistical formula for building the confidence interval.*\n\n"
                "- **Wilson (default):** balanced choice\n"
                "- **Exact:** conservative, stricter bounds\n"
                "- **Agresti–Coull:** approximate, works well\n"
                "- **Normal:** classic textbook formula")
    with colM2:
        use_fpc_qa = st.checkbox("Apply FPC (QA pool)", value=True)
        st.caption("🧮 *Finite Population Correction.*\n\n"
                "Useful when the QA pool isn’t huge.\nExample: If you review 300k calls from a pool of 25M, "
                "the FPC effect is minimal, but for smaller pools it matters.")
    with colM3:
        trans_min_in = st.number_input("Transcription min", value=transcription_min, step=0.0001, format="%.4f")
        trans_max_in = st.number_input("Transcription max", value=transcription_max, step=0.0001, format="%.4f")
        st.caption("💵 *Cost range for human transcription (per minute).*")
    ai_unit_in = st.number_input("AI unit cost", value=ai_cost, step=0.0005, format="%.4f")
    st.caption("🤖 *Cost per unit for AI-based transcription/analysis.*")


    # Compute required n and apply FPC against actual QA pool size
    n_req_qa = required_sample_size_ci(p=p_qa, E=E_qa, confidence=conf_qa, method=method_qa)
    n_adj_qa = apply_fpc(n_req_qa, qa_count) if use_fpc_qa else n_req_qa

    # Cost estimates for chosen n
    cmin_real, cmax_real = cost_per_call_range(trans_min_in, trans_max_in, ai_unit_in)
    total_min_real = total_cost(n_adj_qa, cmin_real)
    total_max_real = total_cost(n_adj_qa, cmax_real)

    # Achieved confidence at the selected n
    conf_at_n = achieved_confidence_from_n(n_adj_qa, p_qa, E_qa, method=method_qa)

    st.markdown("#### Results (based on your current month’s QA pool)")
    st.write(f"- Required n (no FPC): **{n_req_qa:,}**")
    st.write(f"- Required n (with FPC vs QA pool = {qa_count:,}): **{n_adj_qa:,}**")
    st.write(f"- Achieved confidence at n (should be ≥ target): **{conf_at_n:.3f}** (target = {conf_qa:.3f})")
    st.write(f"- Per-call cost range: **${cmin_real:.6f} — ${cmax_real:.6f}**")
    st.write(f"- **Estimated QA sampling cost**: **${total_min_real:,.2f} — ${total_max_real:,.2f}**")

    # Confidence curve around the selected n
    st.markdown("#### Confidence Curve (around selected sample size)")
    window = max(5000, int(0.5 * n_adj_qa))
    ns_win = np.arange(max(50, n_adj_qa - window), n_adj_qa + window + 1, max(50, window // 80))
    confs_win = [achieved_confidence_from_n(int(n), p_qa, E_qa, method=method_qa) for n in ns_win]

    figc, axc = plt.subplots()
    axc.plot(ns_win, confs_win)
    axc.axhline(conf_qa, linestyle='--', label='Target confidence')
    axc.axvline(n_adj_qa, linestyle='--', label=f'n* (={n_adj_qa:,})')
    axc.set_xlabel("Sample size (n)")
    axc.set_ylabel("Achieved confidence")
    axc.set_title(f"Confidence vs n (method={method_qa}, p={p_qa}, E=±{E_qa})")
    axc.legend()
    axc.grid(True)
    st.pyplot(figc)

    # Cost curve in the same window
    st.markdown("#### Cost Curve (min/max) for QA Sampling")
    totals_min_win = ns_win * cmin_real
    totals_max_win = ns_win * cmax_real

    figk, axk = plt.subplots()
    axk.plot(ns_win, totals_min_win, label="Lower bound cost")
    axk.plot(ns_win, totals_max_win, label="Upper bound cost")
    axk.axvline(n_adj_qa, linestyle='--', label=f'n* (={n_adj_qa:,})')
    axk.set_xlabel("Sample size (n)")
    axk.set_ylabel("Total variable cost (USD)")
    axk.set_title("Cost vs n (per-call range)")
    axk.legend()
    axk.grid(True)
    st.pyplot(figk)

    # Optional data preview & download
    with st.expander("Preview filtered data (first 500 rows)"):
        st.dataframe(df.head(500), use_container_width=True)

    # Provide merged file download (filtered to >=30s)
    @st.cache_data
    def to_csv_bytes(_df: pd.DataFrame) -> bytes:
        return _df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered merged dataset (CSV)",
        data=to_csv_bytes(df),
        file_name="merged_calls_filtered_30s.csv",
        mime="text/csv",
    )

# ---------------------------
# Recommendations (verbatim)
# ---------------------------
st.header("Recommendations")
st.markdown("""
**1) Process 100% of Sales/Appointment Calls.**  
These directly affect revenue and customer experience; full coverage is justified.

**2) Use statistically justified sampling for QA calls.**  
- Decide acceptable margin of error (±E) and confidence level in business terms.  
- Use Wilson or Exact (Clopper–Pearson) methods to size the sample without relying on normality.  
- Apply Finite Population Correction (FPC) when the QA sample is a noticeable fraction of the QA pool.

**3) Optimize costs with incremental rollout.**  
- Start with the required sample for ±E at 95% confidence; monitor the achieved confidence and widen/narrow E if needed.  
- Increase sample size only if observed variance or drift requires tighter precision.

**4) Keep costs minimal with our infrastructure choices.**  
- Transcription runs on efficient open-source models using **spot instances** (cheapest viable compute that meets demand).  
- Even at low unit costs, analyzing **every** QA call provides minimal extra insight relative to cost; sampling captures the signal efficiently.
""")
