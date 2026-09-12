
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
import streamlit as st

st.set_page_config(
    page_title="Insurance Statistical Diagnostics & Modeling Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Palette
st.markdown("""
<style>
    .reportview-container { background: #fafbfc; }
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-pass { color: #16a34a; font-weight: 700; }
    .status-fail { color: #dc2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    try:
        data = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        data = pd.read_csv(url)
    data["log_charges"] = np.log(data["charges"])
    return data

df = load_dataset()

# ----------------------------------------------------
# Sidebar: Population Stratification Controls
# ----------------------------------------------------
st.sidebar.markdown("### 🔍 Population Cohort Controls")
age_bounds = st.sidebar.slider(
    "Age Horizon",
    int(df["age"].min()), int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)
bmi_bounds = st.sidebar.slider(
    "BMI Horizon",
    float(df["bmi"].min()), float(df["bmi"].max()),
    (float(df["bmi"].min()), float(df["bmi"].max()))
)
selected_smoker = st.sidebar.multiselect("Smoking Status", df["smoker"].unique(), default=list(df["smoker"].unique()))
selected_region = st.sidebar.multiselect("Regions", df["region"].unique(), default=list(df["region"].unique()))

filtered = df[
    (df["age"].between(age_bounds[0], age_bounds[1])) &
    (df["bmi"].between(bmi_bounds[0], bmi_bounds[1])) &
    (df["smoker"].isin(selected_smoker)) &
    (df["region"].isin(selected_region))
]

st.title("🧬 Econometric & Biostatistical Inference Engine")
st.markdown("*Applied Statistical Modeling, Gauss-Markov Validation, and Parametric Diagnostics*")

tab_eda, tab_hyp, tab_reg = st.tabs([
    "📊 Empirical Exploration",
    "🧪 Inferential Hypothesis Testing",
    "⚙️ OLS Modeling, Transformations & Diagnostics"
])

# ----------------------------------------------------
# TAB 1: EMPIRICAL EXPLORATION
# ----------------------------------------------------
with tab_eda:
    if len(filtered) == 0:
        st.warning("No records match the active cohort filters. Adjust your sidebar bounds.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample Cohort", f"{len(filtered):,} records", delta=f"{len(filtered)-len(df)} from total")
        c2.metric("Mean Cost", f"${filtered['charges'].mean():,.2f}")
        c3.metric("Cost IQR (Dispersion)", f"${(filtered['charges'].quantile(0.75) - filtered['charges'].quantile(0.25)):,.2f}")
        c4.metric("Fisher Skewness", f"{filtered['charges'].skew():.2f}")

        st.markdown("---")
        left_col, right_col = st.columns([1.1, 1])

        with left_col:
            st.markdown("**Empirical Moments Summary**")
            num_feats = ["age", "bmi", "children", "charges"]
            stats_data = []
            for feat in num_feats:
                s = filtered[feat]
                stats_data.append({
                    "Metric": feat,
                    "Mean": s.mean(),
                    "Std Dev": s.std(),
                    "Median": s.median(),
                    "IQR": s.quantile(0.75) - s.quantile(0.25),
                    "Skew": s.skew(),
                    "Kurtosis": s.kurtosis()
                })
            st.dataframe(
                pd.DataFrame(stats_data).set_index("Metric").style.format("{:.2f}").background_gradient(cmap="Purples", subset=["Skew", "Kurtosis"]),
                use_container_width=True
            )

            st.markdown("**Bivariate Correlation Heatmap**")
            corr = filtered[num_feats].corr()
            fig_c = px.imshow(corr, text_auto=".3f", color_continuous_scale="Purples", zmin=-1, zmax=1)
            fig_c.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
            st.plotly_chart(fig_c, use_container_width=True)

        with right_col:
            st.markdown("**Density Distribution & Right-Tail Infiltration**")
            fig_kde = px.histogram(
                filtered, x="charges", color="smoker", marginal="violin",
                nbins=35, barmode="overlay", opacity=0.7,
                color_discrete_map={"yes": "#e11d48", "no": "#2563eb"}
            )
            fig_kde.update_layout(height=480, legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_kde, use_container_width=True)

# ----------------------------------------------------
# TAB 2: HYPOTHESIS TESTING LAB
# ----------------------------------------------------
with tab_hyp:
    st.subheader("Automated Hypothesis Verification (alpha = 0.05)")
    
    st.markdown("#### Test 1: Two-Group Comparison with Assumption Verification")
    col_sel, col_out = st.columns([1, 2])
    
    with col_sel:
        test1_target = st.selectbox("Stratification Factor", ["smoker", "sex"])
        test1_metric = st.selectbox("Continuous Target Metric", ["charges", "bmi"])
        g_unique = df[test1_target].unique()
        s1 = df[df[test1_target] == g_unique[0]][test1_metric].dropna()
        s2 = df[df[test1_target] == g_unique[1]][test1_metric].dropna()
        
        # Assumption checks
        p_norm1 = stats.shapiro(s1).pvalue
        p_norm2 = stats.shapiro(s2).pvalue
        p_lev = stats.levene(s1, s2).pvalue
        
        st.markdown("**Assumption Diagnostics:**")
        st.write(f"- Normality ({g_unique[0]}): `p = {p_norm1:.2e}`")
        st.write(f"- Normality ({g_unique[1]}): `p = {p_norm2:.2e}`")
        st.write(f"- Levene's Equal Variance: `p = {p_lev:.2e}`")

    with col_out:
        norm_valid = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        if norm_valid:
            eq_var = p_lev > 0.05
            stat_val, p_val = stats.ttest_ind(s1, s2, equal_var=eq_var)
            engine_str = f"Student's Two-Sample t-test (equal_var={eq_var})"
        else:
            stat_val, p_val = stats.mannwhitneyu(s1, s2, alternative='two-sided')
            engine_str = "Mann-Whitney U Rank-Sum Test (Non-Parametric Fallback)"

        st.info(f"**Applied Engine:** {engine_str}")
        m1, m2 = st.columns(2)
        m1.metric("Calculated Test Statistic", f"{stat_val:,.4f}")
        m2.metric("p-value", f"{p_val:.4e}", delta="Significant" if p_val < 0.05 else "Not Significant")

        if p_val < 0.05:
            st.error(f"**Statistical Decision:** Reject H0. Significant distributional divergence across {test1_target} tiers.")
        else:
            st.success("**Statistical Decision:** Fail to Reject H0. No evidence of significant distributional divergence.")

    st.markdown("---")
    st.markdown("#### Test 2: Categorical Independence or Multi-Factor Variance")
    hyp2_mode = st.radio("Execute:", ["Chi-Square Test of Independence", "One-Way ANOVA with Post-Hoc Tukey HSD"], horizontal=True)

    if "Chi-Square" in hyp2_mode:
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            cx = st.selectbox("Factor A", ["smoker", "sex"])
            cy = st.selectbox("Factor B", ["region", "children"])
            ct = pd.crosstab(df[cx], df[cy])
            c_stat, p_c, df_c, _ = stats.chi2_contingency(ct)
        with t_col2:
            st.dataframe(ct, use_container_width=True)
            st.write(f"**Chi-Square Statistic:** `{c_stat:.4f}` | **df:** `{df_c}` | **p-value:** `{p_c:.4e}`")
            if p_c < 0.05:
                st.error("Reject H0: Significant dependence between factors.")
            else:
                st.success("Fail to Reject H0: Factors demonstrate statistical independence.")
    else:
        groups = [g["charges"].values for _, g in df.groupby("region")]
        f_stat, p_a = stats.f_oneway(*groups)
        
        st.write(f"**One-Way ANOVA:** F = `{f_stat:.4f}`, p-value = `{p_a:.4e}`")
        if p_a < 0.05:
            st.error("Reject H0: Significant between-group variance exists across regions.")
            st.markdown("**Post-Hoc Tukey Honestly Significant Difference (HSD) Multi-Comparison:**")
            tukey = pairwise_tukeyhsd(endog=df["charges"], groups=df["region"], alpha=0.05)
            st.dataframe(pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]), use_container_width=True)
        else:
            st.success("Fail to Reject H0: No significant variance detected across regions.")

# ----------------------------------------------------
# TAB 3: OLS MODELING, TRANSFORMATIONS & DIAGNOSTICS
# ----------------------------------------------------
with tab_reg:
    st.subheader("Statistical Specification & Diagnostics")

    c_opt1, c_opt2 = st.columns([1, 1])
    with c_opt1:
        target_mode = st.radio("Dependent Target Formulation (Y):", ["Raw Charges ($)", "Log-Transformed charges: log(charges)"], horizontal=True)
    with c_opt2:
        model_type = st.radio("Model Structure:", ["Additive Model", "Interaction Model (BMI × Smoker)"], horizontal=True)

    y_var = "log_charges" if "Log" in target_mode else "charges"
    
    if "Interaction" in model_type:
        formula = f"{y_var} ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)"
    else:
        formula = f"{y_var} ~ age + bmi + children + C(sex) + C(smoker) + C(region)"

    ols_model = ols(formula, data=df).fit()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Adjusted R²", f"{ols_model.rsquared_adj:.4f}")
    k2.metric("AIC", f"{ols_model.aic:,.1f}")
    k3.metric("BIC", f"{ols_model.bic:,.1f}")
    k4.metric("F-Test p-value", f"{ols_model.f_pvalue:.2e}")

    resids = ols_model.resid
    fitted = ols_model.fittedvalues

    st.markdown("---")
    st.markdown("#### Gauss-Markov Assumption Health Auditor")
    
    # Tests
    jb_val, jb_p, _, _ = sm.stats.jarque_bera(resids)
    bp_test = sm.stats.diagnostic.het_breuschpagan(resids, ols_model.model.exog)
    bp_p = bp_test[1]

    # Continuous VIF check
    cont_x = sm.add_constant(df[["age", "bmi", "children"]])
    max_vif = max([variance_inflation_factor(cont_x.values, i) for i in range(1, cont_x.shape[1])])

    aud1, aud2, aud3 = st.columns(3)
    with aud1:
        status_h = "✅ PASS" if bp_p > 0.05 else "⚠️ VIOLATION"
        st.markdown(f"**Homoscedasticity (Breusch-Pagan):** {status_h}")
        st.caption(f"BP p-value: `{bp_p:.2e}`")

    with aud2:
        status_n = "✅ PASS" if jb_p > 0.05 else "⚠️ VIOLATION"
        st.markdown(f"**Residual Normality (Jarque-Bera):** {status_n}")
        st.caption(f"JB p-value: `{jb_p:.2e}`")

    with aud3:
        status_v = "✅ PASS" if max_vif < 5.0 else "⚠️ MULTICOLLINEAR"
        st.markdown(f"**Multicollinearity (Max Continuous VIF):** {status_v}")
        st.caption(f"Max VIF: `{max_vif:.2f}` (Safe threshold < 5.0)")

    st.markdown("---")
    d1, d2, d3 = st.columns(3)
    
    with d1:
        fig_rvf = px.scatter(x=fitted, y=resids, opacity=0.5, labels={"x": "Fitted Values", "y": "Residuals"}, title="Residuals vs. Fitted")
        fig_rvf.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rvf, use_container_width=True)

    with d2:
        qq_res = stats.probplot(resids, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq_res[0][0], y=qq_res[0][1], mode='markers', opacity=0.5, name='Residuals'))
        fig_qq.add_trace(go.Scatter(x=qq_res[0][0], y=qq_res[1][1] + qq_res[1][0]*qq_res[0][0], mode='lines', line=dict(color='red', dash='dash'), name='Normal Reference'))
        fig_qq.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    with d3:
        infl = OLSInfluence(ols_model)
        cooks_d = infl.cooks_distance[0]
        fig_cook = px.scatter(x=np.arange(len(cooks_d)), y=cooks_d, labels={"x": "Observation Index", "y": "Cook's Distance"}, title="Cook's Distance (Influence Analysis)")
        fig_cook.add_hline(y=4/len(df), line_dash="dash", line_color="orange")
        st.plotly_chart(fig_cook, use_container_width=True)

    with st.expander("📄 View Full OLS Regression Table"):
        st.text(ols_model.summary().as_text())

    st.markdown("---")
    st.markdown("#### Real-Time Inference & Interval Estimation")
    
    pi1, pi2, pi3, pi4 = st.columns(4)
    with pi1:
        in_a = st.slider("Patient Age", 18, 65, 32)
        in_s = st.selectbox("Sex Tier", ["male", "female"])
    with pi2:
        in_b = st.number_input("Body Mass Index", 15.0, 55.0, 27.5, step=0.1)
        in_c = st.slider("Dependents Count", 0, 5, 1)
    with pi3:
        in_sm = st.selectbox("Smoker Tier", ["no", "yes"])
        in_rg = st.selectbox("Region Tier", list(df["region"].unique()))

    pred_frame = pd.DataFrame({
        "age": [in_a], "sex": [in_s], "bmi": [in_b],
        "children": [in_c], "smoker": [in_sm], "region": [in_rg]
    })
    
    res_obj = ols_model.get_prediction(pred_frame).summary_frame(alpha=0.05)
    
    if "Log" in target_mode:
        # Corrected lognormal expectation: E[Y] = exp(mu + sigma^2 / 2)
        s2 = ols_model.scale
        est_val = np.exp(res_obj["mean"].iloc[0] + s2 / 2.0)
        ci_l = np.exp(res_obj["mean_ci_lower"].iloc[0])
        ci_u = np.exp(res_obj["mean_ci_upper"].iloc[0])
        pi_l = np.exp(res_obj["obs_ci_lower"].iloc[0])
        pi_u = np.exp(res_obj["obs_ci_upper"].iloc[0])
    else:
        est_val = res_obj["mean"].iloc[0]
        ci_l = res_obj["mean_ci_lower"].iloc[0]
        ci_u = res_obj["mean_ci_upper"].iloc[0]
        pi_l = res_obj["obs_ci_lower"].iloc[0]
        pi_u = res_obj["obs_ci_upper"].iloc[0]

    with pi4:
        st.metric("Point Estimate", f"${est_val:,.2f}")
        st.caption(f"**95% CI (Mean Response):** [${ci_l:,.2f}, ${ci_u:,.2f}]")
        st.caption(f"**95% PI (Individual Response):** [${pi_l:,.2f}, ${pi_u:,.2f}]")