import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.set_page_config(page_title="Medical Insurance Statistical Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Looks for insurance.csv in the current working directory
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()

# ----------------------------------------------------
# Sidebar: Interactive Filtering
# ----------------------------------------------------
st.sidebar.header("Filter Dataset")
age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

smoker_opt = st.sidebar.multiselect(
    "Smoker Status",
    options=df["smoker"].unique(),
    default=df["smoker"].unique()
)

region_opt = st.sidebar.multiselect(
    "Region",
    options=df["region"].unique(),
    default=df["region"].unique()
)

filtered_df = df[
    (df["age"] >= age_range[0]) &
    (df["age"] <= age_range[1]) &
    (df["smoker"].isin(smoker_opt)) &
    (df["region"].isin(region_opt))
]

st.title("Applied Statistical Modeling & Insurance Analytics")

tab1, tab2, tab3 = st.tabs([
    "1. Exploratory Data Analysis",
    "2. Hypothesis Testing Lab",
    "3. Live Prediction & Diagnostics"
])

# ----------------------------------------------------
# Tab 1: Exploratory Data Analysis
# ----------------------------------------------------
with tab1:
    st.subheader("Descriptive Metrics")
    num_cols = ["age", "bmi", "children", "charges"]
    
    desc_list = []
    for col in num_cols:
        desc_list.append({
            "Feature": col,
            "Mean": filtered_df[col].mean(),
            "Median": filtered_df[col].median(),
            "Std Dev": filtered_df[col].std(),
            "IQR": filtered_df[col].quantile(0.75) - filtered_df[col].quantile(0.25),
            "Skewness": filtered_df[col].skew(),
            "Kurtosis": filtered_df[col].kurtosis()
        })
    desc_df = pd.DataFrame(desc_list).set_index("Feature")
    st.dataframe(desc_df.style.format("{:.2f}"))

    st.subheader("Visual Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Charges Distribution by Smoker**")
        fig_hist = px.histogram(
            filtered_df, x="charges", color="smoker", marginal="box",
            nbins=40, title="Distribution of Charges"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.write("**BMI vs. Charges (Colored by Smoker)**")
        fig_scatter = px.scatter(
            filtered_df, x="bmi", y="charges", color="smoker",
            hover_data=["age", "region"], title="Charges vs. BMI"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("**Correlation Matrix (Continuous Metrics)**")
    corr = filtered_df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------------------------------------
# Tab 2: Hypothesis Testing Lab
# ----------------------------------------------------
with tab2:
    st.subheader("Hypothesis Test 1: Two-Group Comparison")
    st.markdown("""
    * **H0**: Medical charges are equal between smokers and non-smokers.  
    * **H1**: Medical charges differ significantly between smokers and non-smokers.
    """)
    
    group_smoker = df[df["smoker"] == "yes"]["charges"]
    group_nonsmoker = df[df["smoker"] == "no"]["charges"]

    # Normality Check (Shapiro-Wilk)
    _, p_shapiro_s = stats.shapiro(group_smoker)
    _, p_shapiro_ns = stats.shapiro(group_nonsmoker)
    # Variance Check (Levene's)
    _, p_levene = stats.levene(group_smoker, group_nonsmoker)

    st.write(f"- **Shapiro-Wilk p-value (Smokers)**: `{p_shapiro_s:.4e}`")
    st.write(f"- **Shapiro-Wilk p-value (Non-Smokers)**: `{p_shapiro_ns:.4e}`")
    st.write(f"- **Levene's Test p-value**: `{p_levene:.4e}`")

    # Decision on normality
    is_normal = (p_shapiro_s > 0.05) and (p_shapiro_ns > 0.05)
    
    if is_normal:
        st.info("Both groups are normally distributed. Running Independent Two-Sample t-test.")
        stat, p_val = stats.ttest_ind(group_smoker, group_nonsmoker, equal_var=(p_levene > 0.05))
        test_name = "Two-Sample t-test"
    else:
        st.warning("Normality assumption violated. Executing non-parametric Mann-Whitney U test.")
        stat, p_val = stats.mannwhitneyu(group_smoker, group_nonsmoker)
        test_name = "Mann-Whitney U test"

    st.write(f"**{test_name} Statistic**: `{stat:.4f}` | **p-value**: `{p_val:.4e}`")
    if p_val < 0.05:
        st.error(f"Decision: **Reject H0** at α = 0.05. Smoking status has a statistically significant effect on charges.")
    else:
        st.success(f"Decision: **Fail to Reject H0** at α = 0.05.")

    st.divider()

    st.subheader("Hypothesis Test 2: Categorical Independence & Multi-Group Variance")
    test2_choice = st.radio("Select Test to Inspect:", ["Chi-Square Test (Categorical Link)", "One-Way ANOVA (Charges across Regions)"])

    if test2_choice == "Chi-Square Test (Categorical Link)":
        st.markdown("**H0**: Smoking status is independent of Region.  \n**H1**: Smoking status depends on Region.")
        contingency_table = pd.crosstab(df["smoker"], df["region"])
        st.write("Contingency Table:", contingency_table)
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency_table)
        st.write(f"**Chi-Square**: `{chi2:.4f}` | **Degrees of Freedom**: `{dof}` | **p-value**: `{p_chi2:.4f}`")
        if p_chi2 < 0.05:
            st.error("Decision: **Reject H0** at α = 0.05. Smoking status is dependent on region.")
        else:
            st.success("Decision: **Fail to Reject H0** at α = 0.05. No significant link found between smoking status and region.")

    else:
        st.markdown("**H0**: Mean charges are identical across all 4 regions.  \n**H1**: At least one region differs in mean charges.")
        region_groups = [group["charges"].values for _, group in df.groupby("region")]
        f_stat, p_anova = stats.f_oneway(*region_groups)
        st.write(f"**F-Statistic**: `{f_stat:.4f}` | **p-value**: `{p_anova:.4f}`")
        if p_anova < 0.05:
            st.error("Decision: **Reject H0** at α = 0.05. Regional charges show statistically significant differences.")
        else:
            st.success("Decision: **Fail to Reject H0** at α = 0.05. No significant difference in mean charges across regions.")

# ----------------------------------------------------
# Tab 3: Live Prediction & Diagnostics
# ----------------------------------------------------
with tab3:
    st.subheader("OLS Model Specification & Estimation")

    # Fit Statsmodels OLS
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)"
    model = ols(formula, data=df).fit()

    st.write(f"**R-squared**: `{model.rsquared:.4f}` | **Adjusted R-squared**: `{model.rsquared_adj:.4f}`")
    
    with st.expander("View Full OLS Regression Summary"):
        st.text(model.summary())

    st.subheader("Live Real-Time Cost Prediction")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        in_age = st.slider("Age", 18, 65, 30)
        in_sex = st.selectbox("Sex", ["male", "female"])
    with p_col2:
        in_bmi = st.number_input("BMI", 15.0, 55.0, 25.0, step=0.1)
        in_children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
    with p_col3:
        in_smoker = st.selectbox("Smoker", ["yes", "no"])
        in_region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    input_data = pd.DataFrame({
        "age": [in_age],
        "sex": [in_sex],
        "bmi": [in_bmi],
        "children": [in_children],
        "smoker": [in_smoker],
        "region": [in_region]
    })

    pred_res = model.get_prediction(input_data).summary_frame(alpha=0.05)
    pred_val = pred_res["mean"].iloc[0]
    ci_lower = pred_res["mean_ci_lower"].iloc[0]
    ci_upper = pred_res["mean_ci_upper"].iloc[0]

    st.metric(label="Predicted Charge", value=f"${pred_val:,.2f}")
    st.caption(f"95% Confidence Interval: **[${ci_lower:,.2f}, ${ci_upper:,.2f}]**")

    st.divider()
    st.subheader("Gauss-Markov Diagnostic Checks")

    diag_col1, diag_col2 = st.columns(2)
    residuals = model.resid
    fitted = model.fittedvalues

    with diag_col1:
        st.write("**Residuals vs. Fitted Values** (Linearity & Homoscedasticity)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(fitted, residuals, alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)

    with diag_col2:
        st.write("**Q-Q Plot** (Normality of Residuals)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sm.qqplot(residuals, line='45', fit=True, ax=ax)
        st.pyplot(fig)

    # Multicollinearity (VIF)
    st.write("**Variance Inflation Factor (VIF) for Continuous Predictors**")
    cont_df = df[["age", "bmi", "children"]].copy()
    cont_df_const = sm.add_constant(cont_df)
    vif_data = pd.DataFrame({
        "Feature": cont_df_const.columns,
        "VIF": [variance_inflation_factor(cont_df_const.values, i) for i in range(cont_df_const.shape[1])]
    })
    st.dataframe(vif_data.iloc[1:].style.format({"VIF": "{:.3f}"}))