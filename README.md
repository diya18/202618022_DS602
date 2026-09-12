# Econometric & Biostatistical Inference Engine — Medical Insurance Analytics

**Course:** Statistical Modeling with Python (M.Sc. Data Science)
**Dataset:** Medical Insurance Costs (`insurance.csv`, Kaggle "Medical Cost Personal Datasets")
**Live Application:** [https://202618022ds602-nfu6hqe923atsejk2pn2y9.streamlit.app/](https://202618022ds602-nfu6hqe923atsejk2pn2y9.streamlit.app/)

---

## 1. Dataset Summary

| Field | Type | Description |
|---|---|---|
| `age` | Integer | Age of primary beneficiary (18–64 years) |
| `sex` | Categorical | Biological sex (`male`, `female`) |
| `bmi` | Float | Body Mass Index ($kg/m^2$), objective ratio of body weight to height |
| `children` | Integer | Number of children/dependents covered by health insurance (0–5) |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | US residential area (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | Float | Individual medical costs billed by health insurance (continuous target) |
| `log_charges` | Float | Natural log transformation: $\ln(\text{charges})$ engineered for variance stabilization |

- **1,338 records**, complete with zero missing values.
- **Distributional Profile:** Target `charges` demonstrates strong right-skewness (Fisher skewness $\approx 1.51$) with severe bimodality driven by smoking status stratification. Predictors `age` and `bmi` display quasi-symmetric distributions.

---

## 2. How to Run

### Option A — Local Environment (Recommended)
```bash
# 1. Clone repository and navigate to project folder
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
streamlit run app.py            # Opens at http://localhost:8501
```

### Option B — Google Colab / Remote Cloud
```python
!pip install streamlit
# Upload or write app.py and requirements.txt
!streamlit run app.py & npx localtunnel --port 8501
```

---

## 3. Project Architecture
```
insurance-inference-engine/
├── app.py                     # Streamlit analytics & statistical modeling suite (3 tabs)
├── requirements.txt           # Python dependencies (Streamlit, Statsmodels, Scipy, Plotly, etc.)
├── README.md                  # Statistical synthesis, diagnostics, and project documentation
└── data/
    └── insurance.csv          # Benchmark dataset (auto-download fallback via URL included)
```

---

## 4. Statistical Methodology & Empirical Findings

### Part 1 — Inferential Hypothesis Testing ($\alpha = 0.05$)

* **Two-Group Comparison (Smoking Status on `charges`):**
  * *Normality Diagnostics:* Shapiro-Wilk test strongly rejects normality for both smoker ($p \approx 3.6 \times 10^{-9}$) and non-smoker ($p \approx 1.4 \times 10^{-28}$) cohorts.
  * *Variance Homogeneity:* Levene's test strongly rejects equal variance ($p \approx 1.6 \times 10^{-66}$).
  * *Applied Engine:* Fallback to non-parametric **Mann-Whitney U Rank-Sum Test** ($U \approx 284,133$, $p \approx 5.3 \times 10^{-130}$).
  * *Decision:* **Reject $H_0$**. Smokers incur significantly higher medical expenses than non-smokers.

* **Two-Group Comparison (Sex on `bmi`):**
  * Both cohorts satisfy mild distributional regularity; two-sample testing confirms no statistically significant difference in mean BMI between males and females at $\alpha = 0.05$.

* **Categorical Independence (Chi-Square Contingency Test):**
  * *Smoking Status $\times$ Region:* $\chi^2 \approx 7.34$, $df = 3$, $p \approx 0.062$. **Fail to Reject $H_0$** at $\alpha = 0.05$. Smoking prevalence is statistically independent of US geographic quadrant.
  * *Smoking Status $\times$ Sex:* $\chi^2 \approx 7.76$, $df = 1$, $p \approx 0.0053$. **Reject $H_0$**. Statistically significant dependence between gender and smoking prevalence in this sample.

* **Multi-Factor Variance Analysis (One-Way ANOVA & Tukey HSD):**
  * *Charges across Geographic Regions:* $F \approx 2.97$, $p \approx 0.031$. **Reject $H_0$**. Average charges differ significantly across regions, with post-hoc Tukey HSD identifying elevated billing in the southeast quadrant due to higher BMI concentration.

---

### Part 2 — Ordinary Least Squares (OLS) Modeling & Moderation

The engine evaluates both **Additive** and **Interaction ($\text{BMI} \times \text{Smoker}$)** specifications across raw and log-transformed scales:

#### 1. Baseline Additive Model (`charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)`)
* Explains $\approx 75.1\%$ of total variance ($R^2 = 0.751$, $\text{Adj } R^2 = 0.749$).
* `smoker[T.yes]` contributes a substantial cost premium of $\approx +\$23,848$ ($p < 0.001$).
* Residual diagnostics show pronounced heteroscedasticity and right-skewed non-normality.

#### 2. Moderation / Interaction Model (`charges ~ ... + bmi:C(smoker)`)
* Model explanatory power increases to **$R^2 = 0.841$ (Adjusted $R^2 = 0.840$)**.
* **Key Moderation Finding:** The main effect of BMI for non-smokers is statistically insignificant ($+\$26.8$ per BMI unit, $p = 0.358$). In contrast, the interaction term `bmi:smoker[T.yes]` contributes an additional **$+\$1,435.60$ per BMI unit** ($p < 0.001$). High BMI significantly magnifies financial risk *only* when paired with smoking.

#### 3. Log-Linear Transformation Model (`log_charges ~ ...`)
* Effectively stabilizes error variance across wide charge magnitudes.
* Predictions apply the **Lognormal retransformation expectation adjustment**:
  $$\mathbb{E}[Y \mid X] = \exp\left(\hat{\mu}_{\ln} + \frac{\hat{\sigma}^2}{2}\right)$$
  to eliminate naive geometric mean underestimation bias.

---

### Part 3 — Gauss-Markov Assumption Diagnostics

| Assumption | Diagnostic Engine | Test Statistic / Metric | Health Status | Empirical Takeaway |
|---|---|---|---|---|
| **Homoscedasticity** | Breusch-Pagan Test | $p < 10^{-15}$ (Raw Charges) | ⚠️ Heteroscedastic | Error variance expands with predicted charges; log-transformation noticeably stabilizes residual spread. |
| **Residual Normality** | Jarque-Bera & Normal Q-Q | $JB \approx 600+$, $p < 10^{-50}$ | ⚠️ Violation | Bimodal residuals caused by separated cost strata; asymptotic sample size ensures reliable coefficient estimates. |
| **Multicollinearity** | Variance Inflation Factor (VIF) | Continuous Max VIF $\approx 1.01$ | ✅ PASS | Continuous predictors (`age`, `bmi`, `children`) have VIF $< 1.1$, well below the critical threshold of $5.0$. |
| **Influential Outliers** | Cook's Distance ($D_i$) | Threshold: $4/N \approx 0.003$ | Monitored | Outliers align with rare high-cost catastrophic hospital claims without distorting overall regression slopes. |

---

## 5. Streamlit Dashboard Architecture

| Tab Component | Analytical Modules & Functionality |
|---|---|
| **📊 Tab 1: Empirical Exploration** | Interactive demographic filters (Age, BMI, Smoking, Region); live metric summary cards (cohort size, mean cost, IQR, Fisher skewness); empirical moments data table; bivariate correlation heatmap; dual-density overlay histogram with marginal violin plots. |
| **🧪 Tab 2: Inferential Hypothesis Testing** | Automated pre-test verification (Shapiro-Wilk for normality, Levene's for homoscedasticity); automatic dispatch between Student's $t$-test and Mann-Whitney U test; Chi-Square contingency matrix; One-Way ANOVA with post-hoc Tukey HSD pairwise contrasts. |
| **⚙️ Tab 3: OLS Modeling, Transformations & Diagnostics** | Target specification toggle (Raw vs. $\ln(\text{charges})$); Model structure switch (Additive vs. $\text{BMI} \times \text{Smoker}$ Interaction); full OLS summary table (AIC, BIC, Adjusted $R^2$, $F$-test); Gauss-Markov audit indicators; diagnostic visual plots (Residuals vs. Fitted, Normal Q-Q, Cook's Distance); live interactive predictions with 95% Confidence Intervals and 95% Prediction Intervals. |

---

## 6. Live Application Deployment

> **Streamlit App URL:** [https://202618022ds602-nfu6hqe923atsejk2pn2y9.streamlit.app/](https://202618022ds602-nfu6hqe923atsejk2pn2y9.streamlit.app/)
