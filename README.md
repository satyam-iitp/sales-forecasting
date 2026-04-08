# 🛒 Walmart Retail Sales Demand Forecasting

> **Can we predict how much a Walmart store will sell next week — well enough to save $114 million in inventory costs?**  
> This project answers that question using 2.5 years of weekly sales data across 45 stores.

---

## 📌 Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Critical Bug Found & Fixed](#critical-bug-found--fixed)
- [Business Impact](#business-impact)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Recommendations](#recommendations)

---

## Business Problem

Retailers walk a tightrope every week:

- **Order too much** → unsold goods sit in warehouses (holding costs, spoilage, tied-up capital)
- **Order too little** → empty shelves, lost sales, unhappy customers who may never return

The goal of this project is to build a forecasting system that predicts `Weekly_Sales` accurately enough to minimize both types of error — and quantify exactly how much money better forecasting saves.

---

## Dataset

| Property | Value |
|---|---|
| Source | Walmart Store Sales (Kaggle) |
| Rows | 6,435 |
| Stores | 45 |
| Time Range | Feb 2010 – Oct 2012 |
| Frequency | Weekly |

**Columns:**

| Column | Type | Description |
|---|---|---|
| `Store` | int | Store ID (1–45) |
| `Date` | date | Week-ending date |
| `Weekly_Sales` | float | **Target variable** — total sales that week |
| `Holiday_Flag` | int | 1 = holiday week, 0 = normal week |
| `Temperature` | float | Regional temperature (°F) |
| `Fuel_Price` | float | Local fuel price ($/gallon) |
| `CPI` | float | Consumer Price Index (inflation measure) |
| `Unemployment` | float | Regional unemployment rate (%) |

---

## Project Structure

```
walmart-demand-forecasting/
│
├── data/
│   └── walmart_sales.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Load data, inspect columns, define problem
│   ├── 02_eda.ipynb                  # Visualise trends, seasonality, correlations
│   ├── 03_forecasting.ipynb          # Build & evaluate 3 forecasting models
│   └── 04_cost_analysis.ipynb        # Translate errors into real business costs
│
├── requirements.txt
└── README.md
```

---

## Methodology

### Step 1 — Understand the Data
Inspect shape, data types, null values. Define what one row means, what we're predicting, and what might influence it.

### Step 2 — Exploratory Analysis
Visualise sales trends over time, compare holiday vs non-holiday weeks, look at how each store behaves, and check whether economic factors (CPI, fuel price, unemployment) correlate with sales.

### Step 3 — Forecasting Models
Three models are built in order of sophistication:

| # | Model | Idea |
|---|---|---|
| 1 | Last Week's Sales | "Whatever sold last week will sell this week" |
| 2 | 4-Week Rolling Average | Average of the last 4 weeks |
| 3 | Prophet (per-store) | Facebook's time-series model that learns yearly seasonality |

**Train/test split:** 80% of each store's history for training, the final 20% for testing — applied independently per store (see bug note below).

### Step 4 — Cost Analysis
Errors aren't just accuracy numbers — they cost money. We modelled two cost types:
- **Overstock** (predicted > actual): 10% of the excess (storage + spoilage)
- **Stockout** (predicted < actual): 20% of the gap (lost sales + customer churn)

---

## Key Results

| Model | MAE | MAPE | Total Cost (45 stores, ~10 months) |
|---|---|---|---|
| Last Week Baseline | $638,087 | 61.0% | $127.3M |
| Rolling Avg (4-wk) | $455,910 | 43.6% | $91.3M |
| **Prophet (per-store)** | **$56,214** | **5.4%** | **$12.9M** |

Prophet is **~10× more accurate** than the naive baseline and reduces total inventory cost by **90%**.

---

## Critical Bug Found & Fixed

> ⚠️ A data-split bug in the original code caused severely inflated errors.

**The bug:** The original code did a single global 80/20 split on the full dataset. Because rows were ordered by store number first, stores 1–36 landed entirely in training and stores 37–45 landed entirely in testing. Prophet was asked to predict 9 stores it had **never seen** — so it fell back to the global average (~$1M/week). Some of those stores only sell $300K/week, leading to errors of $700K+ per prediction.

**The fix:** Apply the 80/20 split **within each store independently** — every store contributes both training and test weeks.

```python
# ❌ Wrong — accidentally trains on stores 1-36, tests on stores 37-45
split = int(len(df) * 0.8)
train = df.iloc[:split]
test  = df.iloc[split:]

# ✅ Correct — every store gets its own 80/20 chronological split
train_list, test_list = [], []
for store_id in sorted(df["Store"].unique()):
    store_df = df[df["Store"] == store_id].sort_values("Date")
    sp = int(len(store_df) * 0.8)
    train_list.append(store_df.iloc[:sp])
    test_list.append(store_df.iloc[sp:])

train = pd.concat(train_list)
test  = pd.concat(test_list)
```

---

## Business Impact

Upgrading from the naive Last-Week baseline to the seasonal Prophet model:

| Metric | Value |
|---|---|
| Cost savings over test period | **$114.4M** |
| Annualised savings (45 stores) | **~$141M / year** |
| Estimated implementation cost | $50,000 (one-time) |
| 1-Year ROI | **>100,000%** |

Every 1% improvement in MAPE translates to approximately **$1.8M in annual savings** across 45 stores.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/walmart-demand-forecasting.git
cd walmart-demand-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the data file
# Place walmart_sales.csv inside the data/ folder

# 4. Run notebooks in order
jupyter notebook notebooks/01_data_understanding.ipynb
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `prophet` | Time-series forecasting |
| `scikit-learn` | Evaluation metrics (MAE, RMSE) |

---

## Recommendations

1. **Deploy Prophet** as the operational forecasting baseline immediately.
2. **Prioritise holiday-week accuracy** — these weeks drive the largest error spikes in naive models.
3. **Penalise under-forecasting more** during model tuning (stockouts cost 2× more than overstocking).
4. **Audit the bottom 10 stores** with consistently high forecast errors — they may need custom seasonality calibration.
5. **Add economic regressors** (CPI, fuel price, unemployment) as external inputs to Prophet in the next iteration to push MAPE below 5%.

---

*Analysis period: February 2010 – October 2012 | 45 Walmart stores | Weekly frequency*
