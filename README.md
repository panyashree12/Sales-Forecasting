# Sales Forecasting & Demand Analytics

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Data Analytics and Data Science internship portfolio project designed to model historical retail sales, evaluate temporal patterns and promotional sensitivity, benchmark classical and seasonal time-series models, and deliver an interactive executive Streamlit dashboard.

---

## 1. Project Overview
Demand forecasting is a foundational capability in retail and e-commerce supply chains. This project delivers an end-to-end analytical and statistical forecasting framework built on 4 years of retail transaction data. It covers raw transaction ingestion, data hygiene, exploratory data analysis, time-series decomposition, chronological train/test validation, forecasting using **ARIMA** and **Holt-Winters Exponential Smoothing**, and deployment via an interactive **Streamlit** dashboard.

## 2. Problem Statement
Retailers face substantial operational risks from inaccurate demand forecasting:
* **Under-forecasting:** Results in stockouts, unfulfilled customer demand, lost revenue, and damaged brand loyalty during peak seasonal shopping periods.
* **Over-forecasting:** Accumulates excessive safety stock, inflated warehouse carrying costs, capital lockup, and inventory write-downs through deep liquidation discounting.
* **Margin Erosion:** Discretionary discounting without data-backed guardrails often cannibalizes bottom-line profitability despite driving superficial top-line revenue volume.

This project addresses these challenges by developing a robust, reproducible forecasting framework that models cyclical weekly shopping patterns and seasonal holiday surges while providing risk-quantified confidence intervals for procurement and financial planning.

## 3. Objectives
* **Data Hygiene & Aggregation:** Clean 9,994 multi-year retail transactions, handle missing values and duplicates, and build a continuous daily time series.
* **Trend & Seasonality Decomposition:** Isolate multi-year macro growth trends, annual Q4 holiday demand peaks, and intra-week transaction cycles.
* **Promotional Sensitivity Analysis:** Quantify the true impact of discount percentages on sales volume and net profit margins to identify profitability ceilings.
* **Chronological Model Benchmarking:** Evaluate time-series models (**ARIMA(1,1,1)** vs **Holt-Winters Exponential Smoothing**) on an unseen 20% temporal holdout window without data leakage.
* **Operational Forecast Generation:** Generate a 30-day out-of-sample forward sales forecast with 95% confidence bounds.
* **Executive Decision Support:** Build a multi-page interactive Streamlit dashboard translating mathematical models into operational and merchandising recommendations.

## 4. Dataset Description & Size
* **Source:** Retail Store Transactions Dataset (`data/sales_data.csv`)
* **Total Transactions:** 9,994 orders
* **Dataset Shape:** 9,994 rows × 21 columns
* **Historical Date Span:** January 3, 2014 – December 30, 2017 (4 operating years / 1,458 continuous calendar days)
* **Core Attributes:**
  * `Order Date`, `Ship Date`, `Ship Mode`: Temporal and logistics markers.
  * `Category`, `Sub-Category`, `Product Name`: Merchandising hierarchy across Furniture, Office Supplies, and Technology.
  * `Sales`: Transaction gross revenue (numerical target).
  * `Quantity`: Number of units purchased per order line.
  * `Discount`: Commercial discount rate applied (0.00 to 0.80).
  * `Profit`: Net operating profit generated per transaction line.
  * `Region`, `State`, `City`, `Postal Code`: Geographic distribution.

## 5. Technologies Used
* **Programming Language:** Python 3.10+ (tested on Windows & Python 3.14)
* **Data Processing & Manipulation:** `pandas`, `numpy`
* **Statistical Modeling & Time Series:** `statsmodels` (ARIMA, Holt-Winters Exponential Smoothing, Seasonal Decompose, ADF Test)
* **Machine Learning & Metrics:** `scikit-learn` (MAE, RMSE, train-test splitting)
* **Model Serialization:** `joblib`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Interactive Dashboard:** `streamlit`
* **Interactive Notebook:** `jupyter`, `nbformat`, `ipykernel`

## 6. Project Workflow
```
┌─────────────────────────────────────────────────────────┐
│              1. Data Ingestion & Hygiene                │
│    Load sales_data.csv, parse datetimes, drop nulls     │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│         2. Exploratory Data Analysis & Feature Eng      │
│   Aggregate daily sales, create calendar/promo features │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│      3. Chronological Train-Test Split (80% / 20%)      │
│      1,166 days Training   │   292 days Holdout Test    │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│          4. Model Training & Benchmarking               │
│      ARIMA(1,1,1)   vs   Holt-Winters Exp Smoothing     │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│         5. Champion Selection & Out-of-Sample FC        │
│   Holt-Winters chosen → 30-Day Forecast + 95% Conf Bounds│
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│            6. Interactive Streamlit Deployment          │
│   Executive Overview, Trend, Seasonality, Forecast, EDA  │
└─────────────────────────────────────────────────────────┘
```

## 7. Exploratory Data Analysis (EDA) Performed
1. **Category Performance:** Technology ($836,154) and Furniture ($741,999) constitute the majority of revenue, with Technology generating superior profit margins.
2. **Sub-Category Pareto:** Phones and Chairs account for the highest sales volume, while Tables and Bookcases incur frequent margin deficits.
3. **Macro Trend Dynamics:** Moving average smoothing (7-day and 30-day) confirms steady year-over-year revenue expansion from 2014 through 2017.
4. **Seasonal Cycles:** Additive decomposition highlights strong recurrent weekly waves (period = 7) and massive Q4 surges occurring consistently each November and December.
5. **Day-of-Week Variation:** Tuesday through Friday record the highest order velocity, while Sunday transaction volume declines by ~35%.

## 8. Forecasting Methods
Two classical and robust time-series forecasting approaches were implemented:
1. **ARIMA (Autoregressive Integrated Moving Average):**
   * Configured as `ARIMA(1, 1, 1)` following stationarity validation via the Augmented Dickey-Fuller (ADF) test ($p < 0.05$).
   * Captures first-order autoregressive momentum and moving-average shock persistence.
2. **Holt-Winters Exponential Smoothing:**
   * Configured with additive trend and additive weekly seasonality (`seasonal_periods=7`).
   * Explicitly updates smoothing parameters ($\alpha$, $\beta$, $\gamma$) to model level, growth trajectory, and intra-week trading cadences.

## 9. Model Evaluation Metrics
Models were benchmarked strictly on the **unseen 20% holdout test window** (292 days: March 14, 2017 – December 30, 2017):
* **MAE (Mean Absolute Error):** Measures average dollar magnitude of forecast errors.
* **RMSE (Root Mean Squared Error):** Penalizes larger outlier forecasting discrepancies.
* **MAPE (Mean Absolute Percentage Error):** Evaluated across non-zero sales days to avoid division-by-zero artifacts.

## 10. Final Model Results
Actual evaluation metrics produced by the pipeline:

| Model | MAE ($) | RMSE ($) | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Holt-Winters Exp Smoothing** | **$1,628.40** | **$2,611.96** | **244.54%** | **Champion (Selected)** |
| **ARIMA(1,1,1)** | $1,739.46 | $2,739.40 | 395.32% | Benchmark |

**Key Finding:** Holt-Winters achieved a **$111.06 lower MAE** and **$127.44 lower RMSE** compared to ARIMA. Modeling the 7-day cyclical seasonality provides critical signal in capturing weekday-to-weekend shopping fluctuations.

## 11. Dashboard Description
The interactive Streamlit dashboard (`app.py`) provides an executive-ready user interface structured across 7 dedicated sections:
1. **Executive Overview:** 4 high-level KPI cards (Total Revenue, Active Daily Average, Forecast Horizon, 30-Day Projected Revenue), Category revenue pie chart, and macro historical trajectory.
2. **Sales Trend:** Interactive daily sales line chart with selectable 7/14/30/60-day moving average smoothing, annual bar charts, and quarterly revenue bars.
3. **Seasonality Analysis:** Monthly aggregated sales bar chart, day-of-week distribution, and an annual-monthly seasonality heatmap.
4. **Forecast:** Interactive model toggle (Holt-Winters vs ARIMA), 30-day out-of-sample forward projections with 95% confidence interval ribbons, KPI summary, tabular forecast preview, and one-click CSV download.
5. **Model Performance:** Side-by-side metric comparison table, MAE/RMSE error comparison charts, 7-day smoothed holdout test ground truth vs predictions, and residual diagnostics.
6. **Promotion Analysis:** Discount tier segmentation (0%, 1-15%, 16-30%, 31-50%, 50%+), sales volume vs profitability bar charts, and discount-to-profit margin scatter plots.
7. **Business Insights:** Synthesized operational recommendations covering inventory pre-stocking, promotion governance, and staffing cadence.

## 12. Business Insights & Strategic Recommendations
* **Holiday Pre-Stocking:** November and December experience a **2.4x demand multiplier** over baseline February sales. Supply chain managers should execute purchase orders with suppliers by early September to avoid spot-freight premiums and stockouts.
* **Discretionary Discount Guardrails:** Promotional analysis reveals that discounts between **0% and 20%** drive healthy margins. However, discounts exceeding **20%** systematically result in negative operating profits. Discretionary discount authority should be capped at 20%.
* **Fulfillment Staffing Cadence:** Transactions peak Tuesday through Friday. Fulfillment center shift allocations should prioritize mid-week staffing to preserve 2-day delivery SLAs, utilizing Sunday for warehouse cycle counts and maintenance.
* **Model Retraining Schedule:** Implement a monthly automated pipeline recalibration to incorporate newly closed daily transactions into Holt-Winters parameter updates.

## 13. Project Structure
```
Sales-Forecasting/
│
├── data/
│   └── sales_data.csv                    # Cleaned 9,994-row retail transactions dataset
│
├── models/
│   ├── forecasting_model.pkl             # Serialized model bundle (Joblib)
│   ├── forecast_results.csv              # 30-day forward forecast with confidence bounds
│   └── model_metrics.csv                 # Model evaluation benchmark results
│
├── notebooks/
│   └── sales_forecasting_analysis.ipynb  # Executed, 18-section analytical notebook
│
├── screenshots/                          # Dashboard preview captures
│
├── src/
│   └── train_forecasting_model.py        # Modular training & evaluation pipeline
│
├── app.py                                # Full Streamlit analytics dashboard
├── requirements.txt                      # Project dependency specification
├── README.md                             # Comprehensive project documentation
└── .gitignore                            # Standard Python & OS ignore rules
```

## 14. Installation Instructions

### 1. Clone or Open Project Directory
```bash
cd Sales-Forecasting
```

### 2. Create and Activate Virtual Environment (Recommended)
**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Forecasting Models & Generate Artifacts
```bash
python src/train_forecasting_model.py
```

## 15. How to Run the Jupyter Notebook
To inspect the pre-rendered notebook or re-run each cell interactively:
```bash
jupyter notebook notebooks/sales_forecasting_analysis.ipynb
```
*(Or open directly in VS Code / JupyterLab)*

## 16. How to Run the Streamlit Dashboard
Launch the interactive web application from the project root:
```bash
python -m streamlit run app.py
```
Or simply:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

## 17. Future Improvements
* **Exogenous Regressors (SARIMAX):** Incorporate promotional calendar events, inflation indices, and regional marketing spend as dynamic regressors.
* **Hierarchical Reconciliation:** Implement bottom-up and top-down hierarchical forecasting (e.g., reconciling Store → Category → SKU levels).
* **Machine Learning Ensembling:** Benchmark LightGBM / XGBoost with lagged rolling window features against Holt-Winters.
* **Automated CI/CD Retraining:** Schedule weekly model re-estimation using GitHub Actions or Airflow.

## 18. Author Section
* **Role:** Data Analytics / Data Science Intern
* **Domain:** Retail E-Commerce Sales & Supply Chain Demand Analytics
* **Project Status:** Complete, Validated & Production-Ready
