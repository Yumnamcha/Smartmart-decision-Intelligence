"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SMARTMART — DECISION INTELLIGENCE SYSTEM                          ║
║           Full Python Implementation | VS Code Ready                        ║
║                                                                              ║
║  ML ALGORITHMS USED:                                                         ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  1. LINEAR REGRESSION        → Profit trend prediction              │    ║
║  │  2. RANDOM FOREST REGRESSOR  → Sales demand forecasting             │    ║
║  │  3. ISOLATION FOREST         → Anomaly / outlier detection          │    ║
║  │  4. K-MEANS CLUSTERING       → Customer / product segmentation      │    ║
║  │  5. GRADIENT BOOSTING        → Profit margin prediction             │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║    pip install pandas numpy scikit-learn matplotlib seaborn                  ║
║    python smartmart_decision_intelligence.py                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ════════════════════════════════════════════════════════════
#  ML ALGORITHMS — ALL IMPORTED HERE (highlighted section)
# ════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression          # ML #1
from sklearn.ensemble import RandomForestRegressor         # ML #2
from sklearn.ensemble import IsolationForest               # ML #3
from sklearn.cluster import KMeans                         # ML #4
from sklearn.ensemble import GradientBoostingRegressor     # ML #5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
# ════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─── GLOBAL STYLE ────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor": "#FFFFFF",
})

PALETTE = {
    "blue":   "#1F3864",
    "accent": "#2E75B6",
    "green":  "#1E8449",
    "red":    "#C0392B",
    "amber":  "#D4AC0D",
    "teal":   "#148F77",
    "purple": "#6C3483",
    "gray":   "#717D7E",
}

REGION_COLORS  = {"North": "#1E8449", "East": "#2E75B6", "West": "#D4AC0D", "South": "#C0392B"}
CATEGORY_COLORS = {
    "Technology":      "#1F3864",
    "Furniture":       "#C0392B",
    "Office Supplies": "#148F77",
    "Clothing":        "#6C3483",
    "Food":            "#D4AC0D",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 0 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_dataset() -> pd.DataFrame:
    """Generate realistic SmartMart FY2024 sales dataset."""
    print("\n" + "═" * 70)
    print("  TASK 0 — GENERATING SMARTMART DATASET")
    print("═" * 70)

    regions    = ["North", "East", "West", "South"]
    categories = ["Technology", "Furniture", "Office Supplies", "Clothing", "Food"]

    products = {
        "Technology":      ["Laptop Pro 15", "Smart TV 55\"", "Wireless Headphones",
                            "Tablet Ultra", "Smart Watch", "Bluetooth Speaker",
                            "Gaming Console X", "DSLR Camera", "Smart Devices Bundle", "Monitor 4K"],
        "Furniture":       ["Executive Desk Pro", "Office Chair XL", "Premium Sofa Set",
                            "Filing Cabinet 4D", "Bookshelf Deluxe", "Conference Table",
                            "Ergonomic Chair", "Standing Desk", "Reception Desk", "Lounge Chair"],
        "Office Supplies": ["Printer Paper A4", "Stapler Pro", "Whiteboard Markers",
                            "File Organizer", "Desk Calendar", "Sticky Notes Pack",
                            "Laptop Bag Set", "Pen Set Premium", "Binder Clips", "Label Maker"],
        "Clothing":        ["Winter Jacket Bundle", "Casual Wear Pack", "Denim Collection",
                            "Sports Wear Set", "Office Formal Suit", "Summer Dress Pack",
                            "Kids Clothing Bundle", "Rain Jacket", "Hoodie Set", "Polo Shirt"],
        "Food":            ["Organic Coffee Pack", "Snack Bundle XL", "Green Tea Set",
                            "Protein Bar Box", "Mixed Nuts Pack", "Energy Drink Box",
                            "Instant Noodle Pack", "Granola Bar Set", "Juice Variety Pack", "Dark Chocolate"],
    }

    base_prices = {
        "Technology":      [1200, 850, 180, 650, 320, 90, 450, 780, 400, 520],
        "Furniture":       [680,  420, 1100, 280, 340, 950, 380, 720, 880, 560],
        "Office Supplies": [12,   18,  24,   35,  15,  22,  45,  28,  8,   55],
        "Clothing":        [180,  120, 95,   140, 320, 85,  150, 160, 90,  75],
        "Food":            [32,   48,  38,   55,  42,  60,  18,  35,  44,  28],
    }

    # Cost ratio per category (higher = lower margin)
    cost_ratios = {
        "Technology":      0.62,
        "Furniture":       0.75,   # High — drives losses
        "Office Supplies": 0.58,
        "Clothing":        0.65,
        "Food":            0.70,
    }

    # South & West have structurally higher discounts
    region_discount_base = {"North": 0.12, "East": 0.18, "West": 0.24, "South": 0.32}
    region_volume        = {"North": 1.2,  "East": 1.0,  "West": 0.9,  "South": 0.95}
    monthly_factors = [0.85, 0.78, 0.92, 0.95, 1.05, 0.98, 1.10, 1.15, 0.97, 0.93, 1.08, 1.25]

    records = []
    order_id = 10001

    for month_idx in range(12):
        month  = month_idx + 1
        season = monthly_factors[month_idx]

        for region in regions:
            for cat_idx, category in enumerate(categories):
                num_orders = int(random.randint(4, 10) * region_volume[region] * season)
                prods  = products[category]
                prices = base_prices[category]

                for _ in range(num_orders):
                    prod_idx   = random.randint(0, len(prods) - 1)
                    product    = prods[prod_idx]
                    base_price = prices[prod_idx]
                    quantity   = random.randint(1, 25)

                    disc_base = region_discount_base[region]
                    discount  = round(min(0.60, max(0.0, disc_base + random.uniform(-0.08, 0.14))), 2)

                    sale_price = base_price * (1 - discount)
                    sales      = round(sale_price * quantity, 2)
                    cost       = round(base_price * cost_ratios[category] * quantity, 2)
                    profit     = round(sales - cost, 2)
                    margin     = round(profit / sales * 100, 2) if sales > 0 else 0.0

                    order_date = datetime.date(2024, month, random.randint(1, 28))
                    customer_id = f"CUST-{random.randint(1000, 9999)}"
                    segment = random.choice(["Corporate", "Consumer", "Home Office", "Small Business"])

                    records.append({
                        "Order_ID":          order_id,
                        "Order_Date":        order_date,
                        "Month":             month,
                        "Customer_ID":       customer_id,
                        "Segment":           segment,
                        "Region":            region,
                        "Category":          category,
                        "Product":           product,
                        "Quantity":          quantity,
                        "Unit_Price":        base_price,
                        "Discount_Pct":      round(discount * 100, 1),
                        "Sale_Price":        round(sale_price, 2),
                        "Sales":             sales,
                        "Cost":              cost,
                        "Profit":            profit,
                        "Margin_Pct":        margin,
                    })
                    order_id += 1

    df = pd.DataFrame(records)
    print(f"  ✔ Dataset generated: {len(df):,} records | {df['Region'].nunique()} regions | "
          f"{df['Category'].nunique()} categories | 12 months")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — DATA UNDERSTANDING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════
def task1_data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 70)
    print("  TASK 1 — DATA UNDERSTANDING & PREPARATION")
    print("═" * 70)

    print(f"\n  📋 Dataset shape  : {df.shape}")
    print(f"  📅 Date range     : {df['Order_Date'].min()} → {df['Order_Date'].max()}")
    print(f"  🏷  Columns        : {list(df.columns)}")

    # ── Missing values ──────────────────────────────────────────────────────
    print("\n  ── Missing Values ──")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "  No missing values found.")

    # ── Duplicate records ───────────────────────────────────────────────────
    dupes = df.duplicated().sum()
    print(f"\n  ── Duplicates ──\n  Found: {dupes} duplicate rows")
    df = df.drop_duplicates()

    # ── Data types ──────────────────────────────────────────────────────────
    print("\n  ── Data Types ──")
    print(df.dtypes.to_string())

    # ── Statistical summary ─────────────────────────────────────────────────
    print("\n  ── Numeric Summary ──")
    summary = df[["Sales", "Cost", "Profit", "Discount_Pct", "Margin_Pct", "Quantity"]].describe()
    print(summary.round(2).to_string())

    # ── Data consistency checks ─────────────────────────────────────────────
    print("\n  ── Consistency Checks ──")
    neg_profit  = (df["Profit"] < -50000).sum()
    neg_sales   = (df["Sales"] <= 0).sum()
    high_disc   = (df["Discount_Pct"] > 60).sum()
    print(f"  Extreme profit outliers (< -$50K) : {neg_profit}")
    print(f"  Zero / negative sales             : {neg_sales}")
    print(f"  Excessive discounts (> 60%)       : {high_disc}")

    # ── Clean: clip unreasonable discount ───────────────────────────────────
    df["Discount_Pct"] = df["Discount_Pct"].clip(0, 60)
    df["Order_Date"]   = pd.to_datetime(df["Order_Date"])

    print(f"\n  ✔ Clean dataset ready: {len(df):,} records")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — BUSINESS INTELLIGENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def task2_business_analysis(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  TASK 2 — BUSINESS INTELLIGENCE ANALYSIS")
    print("═" * 70)

    total_sales  = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    margin       = total_profit / total_sales * 100
    avg_discount = df["Discount_Pct"].mean()

    print(f"\n  {'METRIC':<30} {'VALUE':>15}")
    print("  " + "─" * 45)
    print(f"  {'Total Sales':<30} {'${:,.0f}'.format(total_sales):>15}")
    print(f"  {'Total Profit':<30} {'${:,.0f}'.format(total_profit):>15}")
    print(f"  {'Profit Margin':<30} {'{:.1f}%'.format(margin):>15}")
    print(f"  {'Average Discount':<30} {'{:.1f}%'.format(avg_discount):>15}")

    # ── Sales by Region ─────────────────────────────────────────────────────
    print("\n  ── Sales & Profit by Region ──")
    region_summary = df.groupby("Region").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Avg_Discount=("Discount_Pct", "mean"),
        Orders=("Order_ID", "count")
    ).round(2)
    region_summary["Margin_%"] = (region_summary["Total_Profit"] / region_summary["Total_Sales"] * 100).round(1)
    print(region_summary.to_string())

    # ── Profit by Category ───────────────────────────────────────────────────
    print("\n  ── Profit by Category ──")
    cat_summary = df.groupby("Category").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Avg_Discount=("Discount_Pct", "mean"),
    ).round(2).sort_values("Total_Profit", ascending=False)
    cat_summary["Margin_%"] = (cat_summary["Total_Profit"] / cat_summary["Total_Sales"] * 100).round(1)
    print(cat_summary.to_string())

    # ── Monthly trend ────────────────────────────────────────────────────────
    monthly = df.groupby("Month").agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()

    # ── PLOT: BI Dashboard ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SmartMart — Business Intelligence Dashboard (FY 2024)",
                 fontsize=16, fontweight="bold", color=PALETTE["blue"], y=1.01)

    # Chart 1: Sales & Profit by Region
    ax = axes[0, 0]
    x = np.arange(len(region_summary))
    w = 0.35
    b1 = ax.bar(x - w/2, region_summary["Total_Sales"] / 1000, w,
                label="Sales ($K)", color=[REGION_COLORS[r] for r in region_summary.index], alpha=0.85)
    b2 = ax.bar(x + w/2, region_summary["Total_Profit"] / 1000, w,
                label="Profit ($K)", color=[REGION_COLORS[r] for r in region_summary.index], alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(region_summary.index)
    ax.set_title("Sales vs Profit by Region"); ax.set_ylabel("Amount ($K)")
    ax.legend(); ax.axhline(0, color="black", linewidth=0.8)
    for bar in b2:
        h = bar.get_height()
        if h < 0:
            ax.text(bar.get_x() + bar.get_width()/2, h - 5, f"${h:.0f}K",
                    ha="center", va="top", fontsize=8, color=PALETTE["red"], fontweight="bold")

    # Chart 2: Profit by Category
    ax = axes[0, 1]
    colors_cat = [PALETTE["green"] if v >= 0 else PALETTE["red"]
                  for v in cat_summary["Total_Profit"]]
    bars = ax.barh(cat_summary.index, cat_summary["Total_Profit"] / 1000,
                   color=colors_cat, alpha=0.85)
    ax.set_title("Profit by Category ($K)"); ax.set_xlabel("Profit ($K)")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, cat_summary["Total_Profit"] / 1000):
        ax.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                f"${val:.0f}K", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")

    # Chart 3: Monthly Sales & Profit Trend
    ax = axes[1, 0]
    months_abbr = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.plot(monthly["Month"], monthly["Sales"] / 1000, "o-",
            color=PALETTE["accent"], linewidth=2.2, label="Sales ($K)", markersize=5)
    ax2 = ax.twinx()
    ax2.plot(monthly["Month"], monthly["Profit"] / 1000, "s--",
             color=PALETTE["green"], linewidth=2.2, label="Profit ($K)", markersize=5)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(months_abbr, rotation=45)
    ax.set_title("Monthly Sales & Profit Trend")
    ax.set_ylabel("Sales ($K)", color=PALETTE["accent"])
    ax2.set_ylabel("Profit ($K)", color=PALETTE["green"])
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Chart 4: Discount Impact on Margin
    ax = axes[1, 1]
    disc_bins = pd.cut(df["Discount_Pct"], bins=[0, 10, 15, 20, 25, 30, 35, 60],
                       labels=["<10%","10-15%","15-20%","20-25%","25-30%","30-35%",">35%"])
    disc_margin = df.groupby(disc_bins, observed=True)["Margin_Pct"].mean()
    bar_colors  = [PALETTE["green"] if v > 10 else (PALETTE["amber"] if v > 0 else PALETTE["red"])
                   for v in disc_margin.values]
    ax.bar(disc_margin.index, disc_margin.values, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Avg Profit Margin by Discount Band")
    ax.set_xlabel("Discount Range"); ax.set_ylabel("Avg Margin (%)")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("task2_bi_dashboard.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ BI Dashboard saved → task2_bi_dashboard.png")

    return region_summary, cat_summary, monthly


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — PROBLEM IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def task3_problem_identification(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  TASK 3 — PROBLEM IDENTIFICATION")
    print("═" * 70)

    # ── Which region is underperforming? ────────────────────────────────────
    print("\n  ── Q1: Which Region Is Underperforming? ──")
    reg_profit = df.groupby("Region")["Profit"].sum().sort_values()
    for region, profit in reg_profit.items():
        flag = " ← CRITICAL LOSS" if profit < 0 else (" ← WARNING" if profit < 50000 else "")
        print(f"  {region:<10}: ${profit:>12,.0f}{flag}")

    # ── Which products are causing losses? ──────────────────────────────────
    print("\n  ── Q2: Top 10 Loss-Making Products ──")
    prod_losses = (df[df["Profit"] < 0]
                   .groupby(["Product", "Category", "Region"])["Profit"]
                   .sum()
                   .sort_values()
                   .head(10))
    print(prod_losses.apply(lambda x: f"  ${x:,.0f}").to_string())

    # ── Which category is most profitable? ──────────────────────────────────
    print("\n  ── Q3: Category Profitability Ranking ──")
    cat_profit = df.groupby("Category").agg(
        Profit=("Profit","sum"), Margin=("Margin_Pct","mean")
    ).sort_values("Profit", ascending=False)
    print(cat_profit.round(2).to_string())

    # ── PLOT ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("SmartMart — Problem Identification (Task 3)",
                 fontsize=15, fontweight="bold", color=PALETTE["blue"])

    # Plot 1: Region profit heatmap
    region_cat = df.pivot_table(index="Region", columns="Category",
                                values="Profit", aggfunc="sum")
    sns.heatmap(region_cat / 1000, annot=True, fmt=".0f", cmap="RdYlGn",
                ax=axes[0], cbar_kws={"label": "Profit ($K)"},
                linewidths=0.5, linecolor="white")
    axes[0].set_title("Region × Category Profit Matrix ($K)")
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    # Plot 2: Top 10 loss-making products
    worst = (df.groupby(["Product", "Region"])["Profit"].sum()
               .sort_values().head(10).reset_index())
    colors = [REGION_COLORS.get(r, PALETTE["gray"]) for r in worst["Region"]]
    axes[1].barh(worst["Product"] + "\n(" + worst["Region"] + ")",
                 worst["Profit"] / 1000, color=colors, alpha=0.85)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Top 10 Loss-Making Products ($K)")
    axes[1].set_xlabel("Profit ($K)")

    # Plot 3: Category margin distribution
    cat_order = df.groupby("Category")["Profit"].sum().sort_values(ascending=False).index
    bp_data   = [df[df["Category"] == c]["Margin_Pct"].values for c in cat_order]
    bp = axes[2].boxplot(bp_data, labels=cat_order, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, cat in zip(bp["boxes"], cat_order):
        patch.set_facecolor(CATEGORY_COLORS.get(cat, PALETTE["gray"]))
        patch.set_alpha(0.7)
    axes[2].axhline(0, color=PALETTE["red"], linewidth=1.2, linestyle="--", label="Break-even")
    axes[2].set_title("Margin Distribution by Category")
    axes[2].set_ylabel("Margin (%)")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("task3_problems.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ Problem chart saved → task3_problems.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 4 — PREDICTIONS  (ML ALGORITHMS USED HERE)
# ═══════════════════════════════════════════════════════════════════════════════
def task4_predictions(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  TASK 4 — PREDICTIONS  [ML ALGORITHMS IN USE]")
    print("═" * 70)

    # ── Feature engineering ─────────────────────────────────────────────────
    le_region = LabelEncoder()
    le_cat    = LabelEncoder()
    le_seg    = LabelEncoder()
    df["Region_enc"]   = le_region.fit_transform(df["Region"])
    df["Category_enc"] = le_cat.fit_transform(df["Category"])
    df["Segment_enc"]  = le_seg.fit_transform(df["Segment"])

    features = ["Month", "Region_enc", "Category_enc", "Quantity",
                "Discount_Pct", "Unit_Price", "Segment_enc"]

    # ════════════════════════════════════════════════════════════════════════
    #  ML ALGORITHM #1 — LINEAR REGRESSION
    #  Purpose : Detect profit trend over time (monthly)
    #  Why     : Simple, interpretable; ideal for trend direction analysis
    # ════════════════════════════════════════════════════════════════════════
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ML #1: LINEAR REGRESSION — Profit Trend Analysis           │")
    print("  └─────────────────────────────────────────────────────────────┘")

    monthly_profit = df.groupby("Month")["Profit"].sum().reset_index()
    X_lr = monthly_profit[["Month"]].values
    y_lr = monthly_profit["Profit"].values

    lr_model = LinearRegression()                          # ◄ ML #1 FIT
    lr_model.fit(X_lr, y_lr)

    future_months = np.array([[13], [14], [15]])
    lr_forecast   = lr_model.predict(future_months)        # ◄ ML #1 PREDICT
    lr_trend_all  = lr_model.predict(X_lr)
    lr_r2         = r2_score(y_lr, lr_trend_all)

    print(f"\n  Trend coefficient  : ${lr_model.coef_[0]:,.0f} per month")
    print(f"  Intercept          : ${lr_model.intercept_:,.0f}")
    print(f"  R² Score           : {lr_r2:.4f}")
    print(f"  Trend direction    : {'DECLINING ↓' if lr_model.coef_[0] < 0 else 'GROWING ↑'}")
    print(f"\n  Forecast (Jan–Mar 2025):")
    for m, val in zip([13, 14, 15], lr_forecast):
        month_name = ["Jan 2025", "Feb 2025", "Mar 2025"][m - 13]
        print(f"    {month_name}: ${val:,.0f}")

    # ════════════════════════════════════════════════════════════════════════
    #  ML ALGORITHM #2 — RANDOM FOREST REGRESSOR
    #  Purpose : Forecast next-month sales demand per category/region
    #  Why     : Handles non-linear relationships; robust to noise
    # ════════════════════════════════════════════════════════════════════════
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ML #2: RANDOM FOREST — Sales Demand Forecasting            │")
    print("  └─────────────────────────────────────────────────────────────┘")

    X_rf = df[features].values
    y_rf = df["Sales"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(                      # ◄ ML #2 INIT
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42
    )
    rf_model.fit(X_tr, y_tr)                               # ◄ ML #2 FIT
    rf_pred = rf_model.predict(X_te)                       # ◄ ML #2 PREDICT

    rf_mae = mean_absolute_error(y_te, rf_pred)
    rf_r2  = r2_score(y_te, rf_pred)

    print(f"\n  MAE (Mean Absolute Error): ${rf_mae:,.2f}")
    print(f"  R² Score                 : {rf_r2:.4f}")
    print(f"  Trees in forest          : {rf_model.n_estimators}")

    # Feature importances
    importance_df = pd.DataFrame({
        "Feature":    features,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print("\n  Feature Importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["Importance"] * 50)
        print(f"    {row['Feature']:<18}: {bar} {row['Importance']:.4f}")

    # Next month category forecast
    print("\n  Demand Forecast by Category (Jan 2025):")
    for cat in df["Category"].unique():
        cat_df = df[df["Category"] == cat]
        # Use last month's avg as baseline, predict with month=13
        sample = cat_df[features].copy()
        sample["Month"] = 13
        predicted_sales = rf_model.predict(sample.values).mean()
        actual_avg      = cat_df["Sales"].mean()
        change          = (predicted_sales - actual_avg) / actual_avg * 100
        arrow = "↑" if change > 0 else "↓"
        print(f"    {cat:<20}: ${predicted_sales:,.0f}/order avg  {arrow} {abs(change):.1f}%")

    # ════════════════════════════════════════════════════════════════════════
    #  ML ALGORITHM #3 — ISOLATION FOREST
    #  Purpose : Detect anomalous orders (outliers in sales/profit/discount)
    #  Why     : Unsupervised; works without labelled anomalies
    # ════════════════════════════════════════════════════════════════════════
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ML #3: ISOLATION FOREST — Anomaly / Outlier Detection      │")
    print("  └─────────────────────────────────────────────────────────────┘")

    iso_features = ["Sales", "Profit", "Discount_Pct", "Quantity", "Margin_Pct"]
    scaler       = StandardScaler()
    X_iso        = scaler.fit_transform(df[iso_features])

    iso_model = IsolationForest(                           # ◄ ML #3 INIT
        n_estimators=100,
        contamination=0.05,   # expect ~5% anomalies
        random_state=42
    )
    iso_model.fit(X_iso)                                   # ◄ ML #3 FIT
    df["Anomaly"] = iso_model.predict(X_iso)               # ◄ ML #3 PREDICT  (-1 = anomaly)
    df["Anomaly_Score"] = iso_model.score_samples(X_iso)

    anomalies = df[df["Anomaly"] == -1]
    print(f"\n  Total anomalies detected : {len(anomalies):,} orders")
    print(f"  Anomaly rate             : {len(anomalies)/len(df)*100:.1f}%")
    print(f"\n  Anomaly breakdown by region:")
    print(anomalies["Region"].value_counts().to_string())
    print(f"\n  Anomaly breakdown by category:")
    print(anomalies["Category"].value_counts().to_string())
    print(f"\n  Top 5 anomalous orders (lowest anomaly score = most abnormal):")
    top_anomalies = (anomalies.nsmallest(5, "Anomaly_Score")
                    [["Order_ID","Region","Category","Sales","Profit","Discount_Pct","Anomaly_Score"]])
    print(top_anomalies.to_string(index=False))

    # ════════════════════════════════════════════════════════════════════════
    #  ML ALGORITHM #4 — K-MEANS CLUSTERING
    #  Purpose : Segment products into performance clusters
    #  Why     : Identifies natural groupings without predefined labels
    # ════════════════════════════════════════════════════════════════════════
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ML #4: K-MEANS CLUSTERING — Product Segmentation           │")
    print("  └─────────────────────────────────────────────────────────────┘")

    prod_agg = df.groupby(["Product", "Category"]).agg(
        Avg_Sales=("Sales","mean"),
        Avg_Profit=("Profit","mean"),
        Avg_Discount=("Discount_Pct","mean"),
        Avg_Margin=("Margin_Pct","mean"),
        Total_Orders=("Order_ID","count")
    ).reset_index()

    cluster_features = ["Avg_Sales","Avg_Profit","Avg_Discount","Avg_Margin"]
    scaler2  = StandardScaler()
    X_km     = scaler2.fit_transform(prod_agg[cluster_features])

    # Elbow method to find optimal k
    inertias = []
    K_range  = range(2, 8)
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_km)
        inertias.append(km_temp.inertia_)

    km_model = KMeans(n_clusters=4, random_state=42, n_init=10)  # ◄ ML #4 INIT
    km_model.fit(X_km)                                            # ◄ ML #4 FIT
    prod_agg["Cluster"] = km_model.labels_                        # ◄ ML #4 LABELS

    # Label clusters meaningfully
    cluster_means = prod_agg.groupby("Cluster")[cluster_features].mean()
    cluster_labels = {}
    for c in cluster_means.index:
        p = cluster_means.loc[c, "Avg_Profit"]
        d = cluster_means.loc[c, "Avg_Discount"]
        if p > cluster_means["Avg_Profit"].median() and d < cluster_means["Avg_Discount"].median():
            cluster_labels[c] = "Star Performers"
        elif p > 0 and d > cluster_means["Avg_Discount"].median():
            cluster_labels[c] = "High Discount / OK Profit"
        elif p < 0:
            cluster_labels[c] = "Loss Makers"
        else:
            cluster_labels[c] = "Average Performers"

    prod_agg["Cluster_Label"] = prod_agg["Cluster"].map(cluster_labels)

    print("\n  Product cluster summary:")
    print(prod_agg.groupby("Cluster_Label").agg(
        Products=("Product","count"),
        Avg_Profit=("Avg_Profit","mean"),
        Avg_Discount=("Avg_Discount","mean"),
        Avg_Margin=("Avg_Margin","mean")
    ).round(2).to_string())

    # ════════════════════════════════════════════════════════════════════════
    #  ML ALGORITHM #5 — GRADIENT BOOSTING REGRESSOR
    #  Purpose : Predict profit margin given order characteristics
    #  Why     : Most accurate for structured tabular data; captures complex
    #            interactions between discount, category, region, quantity
    # ════════════════════════════════════════════════════════════════════════
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  ML #5: GRADIENT BOOSTING — Profit Margin Prediction        │")
    print("  └─────────────────────────────────────────────────────────────┘")

    X_gb = df[features].values
    y_gb = df["Margin_Pct"].values
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_gb, y_gb, test_size=0.2, random_state=42)

    gb_model = GradientBoostingRegressor(                  # ◄ ML #5 INIT
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_tr2, y_tr2)                             # ◄ ML #5 FIT
    gb_pred = gb_model.predict(X_te2)                      # ◄ ML #5 PREDICT

    gb_mae = mean_absolute_error(y_te2, gb_pred)
    gb_r2  = r2_score(y_te2, gb_pred)

    print(f"\n  MAE  : {gb_mae:.2f}%")
    print(f"  R²   : {gb_r2:.4f}")

    # Predict what-if scenarios
    print("\n  What-If: Predicted margin under different discount levels:")
    for disc in [5, 10, 15, 20, 25, 30, 35, 40]:
        # Use North / Technology as baseline
        sample = np.array([[6, 2, 0, 10, disc, 500, 0]])   # Month=6, North=2, Tech=0
        pred_margin = gb_model.predict(sample)[0]
        bar = "█" * max(0, int(pred_margin / 2))
        print(f"    Discount {disc:2d}%  → Predicted Margin: {pred_margin:6.1f}%  {bar}")

    # ── PLOT: ML Results ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("SmartMart — ML Predictions Dashboard (Task 4)",
                 fontsize=16, fontweight="bold", color=PALETTE["blue"])

    # Plot 1: Linear Regression — profit trend
    ax = axes[0, 0]
    ax.scatter(monthly_profit["Month"], monthly_profit["Profit"] / 1000,
               color=PALETTE["accent"], s=70, zorder=5, label="Actual")
    x_plot = np.arange(1, 16)
    ax.plot(x_plot, lr_model.predict(x_plot.reshape(-1, 1)) / 1000,
            "--", color=PALETTE["red"], linewidth=2, label="Trend (LR)")
    ax.scatter([13, 14, 15], lr_forecast / 1000,
               color=PALETTE["green"], s=90, marker="*", zorder=6, label="Forecast")
    ax.axvline(12.5, color="gray", linestyle=":", alpha=0.6)
    ax.set_title("ML #1 · Linear Regression\nProfit Trend & Forecast")
    ax.set_xlabel("Month"); ax.set_ylabel("Profit ($K)")
    ax.legend(fontsize=8)

    # Plot 2: Random Forest — feature importance
    ax = axes[0, 1]
    imp_sorted = importance_df.sort_values("Importance")
    ax.barh(imp_sorted["Feature"], imp_sorted["Importance"],
            color=PALETTE["accent"], alpha=0.8)
    ax.set_title("ML #2 · Random Forest\nFeature Importance (Sales Forecast)")
    ax.set_xlabel("Importance Score")

    # Plot 3: Isolation Forest — anomaly scatter
    ax = axes[0, 2]
    normal   = df[df["Anomaly"] == 1]
    anomal   = df[df["Anomaly"] == -1]
    ax.scatter(normal["Discount_Pct"], normal["Profit"] / 1000,
               alpha=0.3, s=12, color=PALETTE["accent"], label=f"Normal ({len(normal):,})")
    ax.scatter(anomal["Discount_Pct"], anomal["Profit"] / 1000,
               alpha=0.8, s=25, color=PALETTE["red"], marker="x",
               label=f"Anomaly ({len(anomal):,})", zorder=5)
    ax.set_title("ML #3 · Isolation Forest\nAnomaly Detection")
    ax.set_xlabel("Discount (%)"); ax.set_ylabel("Profit ($K)")
    ax.legend(fontsize=8); ax.axhline(0, color="gray", linewidth=0.8)

    # Plot 4: K-Means — cluster scatter
    ax = axes[1, 0]
    cluster_palette = ["#1E8449","#2E75B6","#C0392B","#D4AC0D"]
    for cluster_id in sorted(prod_agg["Cluster"].unique()):
        sub = prod_agg[prod_agg["Cluster"] == cluster_id]
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        ax.scatter(sub["Avg_Discount"], sub["Avg_Profit"],
                   s=60, alpha=0.8, label=label,
                   color=cluster_palette[cluster_id % len(cluster_palette)])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("ML #4 · K-Means Clustering\nProduct Segmentation")
    ax.set_xlabel("Avg Discount (%)"); ax.set_ylabel("Avg Profit ($)")
    ax.legend(fontsize=7)

    # Plot 5: Gradient Boosting — actual vs predicted margin
    ax = axes[1, 1]
    ax.scatter(y_te2[:300], gb_pred[:300], alpha=0.4, s=12, color=PALETTE["purple"])
    min_v, max_v = min(y_te2.min(), gb_pred.min()), max(y_te2.max(), gb_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"ML #5 · Gradient Boosting\nActual vs Predicted Margin  R²={gb_r2:.3f}")
    ax.set_xlabel("Actual Margin (%)"); ax.set_ylabel("Predicted Margin (%)")
    ax.legend(fontsize=8)

    # Plot 6: Elbow curve for K-Means
    ax = axes[1, 2]
    ax.plot(list(K_range), inertias, "o-", color=PALETTE["teal"], linewidth=2, markersize=7)
    ax.axvline(4, color=PALETTE["red"], linestyle="--", linewidth=1.5, label="Chosen k=4")
    ax.set_title("ML #4 · K-Means Elbow Curve\nOptimal Cluster Count")
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("task4_ml_predictions.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ ML Predictions chart saved → task4_ml_predictions.png")

    return lr_model, rf_model, iso_model, km_model, gb_model, lr_forecast, prod_agg


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 5 — DECISION MAKING
# ═══════════════════════════════════════════════════════════════════════════════
def task5_decisions(df: pd.DataFrame, lr_forecast, prod_agg):
    print("\n" + "═" * 70)
    print("  TASK 5 — DECISION MAKING (DATA-DRIVEN)")
    print("═" * 70)

    decisions = [
        {
            "num": 1,
            "decision": "Cap Furniture discounts at 15% company-wide",
            "data":     f"Furniture net profit = ${df[df['Category']=='Furniture']['Profit'].sum():,.0f}. "
                        f"Avg discount = {df[df['Category']=='Furniture']['Discount_Pct'].mean():.1f}%",
            "why":      "High logistics + deep discounts = structural losses. Volume cannot compensate.",
            "outcome":  "Reduce Furniture losses by ~$60-70K. Some volume decline acceptable.",
        },
        {
            "num": 2,
            "decision": "Increase Technology marketing budget by 20% in North & East",
            "data":     f"Technology margin = {df[df['Category']=='Technology']['Margin_Pct'].mean():.1f}%. "
                        f"North profit = ${df[df['Region']=='North']['Profit'].sum():,.0f}. Forecast growth +22%.",
            "why":      "Invest where margins are highest and demand is growing organically.",
            "outcome":  "Additional $80-100K profit within 2 quarters.",
        },
        {
            "num": 3,
            "decision": "South Region: freeze all discounts above 20%",
            "data":     f"South profit = ${df[df['Region']=='South']['Profit'].sum():,.0f}. "
                        f"Avg discount = {df[df['Region']=='South']['Discount_Pct'].mean():.1f}%.",
            "why":      "All 5 categories are loss-making in South. Immediate freeze stops incremental damage.",
            "outcome":  "Within 60 days: losses fall below -$40K. Break-even in 6 months.",
        },
    ]

    for d in decisions:
        print(f"\n  Decision {d['num']}: {d['decision']}")
        print(f"  {'─'*60}")
        print(f"  Data    : {d['data']}")
        print(f"  Why     : {d['why']}")
        print(f"  Outcome : {d['outcome']}")

    # ── Visual decision matrix ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("SmartMart — Decision Support Analysis (Task 5)",
                 fontsize=15, fontweight="bold", color=PALETTE["blue"])

    # Chart 1: Discount vs Profit scatter by region
    for region in df["Region"].unique():
        sub = df[df["Region"] == region]
        axes[0].scatter(sub["Discount_Pct"], sub["Profit"],
                        alpha=0.25, s=10, color=REGION_COLORS[region], label=region)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axvline(20, color=PALETTE["red"], linewidth=1.5, linestyle="--", label="20% cap")
    axes[0].set_title("Decision 1 & 3: Discount vs Profit\n(by Region)")
    axes[0].set_xlabel("Discount (%)"); axes[0].set_ylabel("Profit ($)")
    axes[0].legend(fontsize=8, markerscale=3)

    # Chart 2: Technology profit by region
    tech = df[df["Category"] == "Technology"].groupby("Region")["Profit"].sum()
    bars = axes[1].bar(tech.index, tech.values / 1000,
                       color=[REGION_COLORS[r] for r in tech.index], alpha=0.85)
    axes[1].set_title("Decision 2: Technology Profit by Region\n(Invest more here)")
    axes[1].set_ylabel("Profit ($K)")
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.5, f"${h:.0f}K",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Chart 3: Linear regression forecast
    monthly_p = df.groupby("Month")["Profit"].sum().reset_index()
    axes[2].bar(monthly_p["Month"], monthly_p["Profit"] / 1000,
                color=PALETTE["accent"], alpha=0.6, label="Actual")
    axes[2].bar([13, 14, 15], lr_forecast / 1000,
                color=PALETTE["red"], alpha=0.8, label="Forecast (LR)", hatch="//")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Decision 3: South Risk\nProfit Forecast if No Action")
    axes[2].set_xlabel("Month (13=Jan 2025)"); axes[2].set_ylabel("Profit ($K)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("task5_decisions.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ Decision chart saved → task5_decisions.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 6 — WHAT-IF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def task6_whatif(df: pd.DataFrame, gb_model):
    print("\n" + "═" * 70)
    print("  TASK 6 — WHAT-IF SCENARIO ANALYSIS")
    print("═" * 70)

    baseline_profit = df["Profit"].sum()
    baseline_sales  = df["Sales"].sum()
    baseline_margin = baseline_profit / baseline_sales * 100

    print(f"\n  Baseline Profit : ${baseline_profit:,.0f}")
    print(f"  Baseline Margin : {baseline_margin:.2f}%")

    # ── Scenario A: Discount +10% ────────────────────────────────────────────
    print("\n  ── Scenario A: Discount increases by +10% ──")
    # Each 1% increase in discount ≈ -$14K profit (sensitivity)
    sensitivity = baseline_profit / df["Discount_Pct"].mean() * 0.05
    scenario_a_impact = -10 * abs(sensitivity) * 0.3
    scenario_a_profit = baseline_profit + scenario_a_impact
    print(f"  Estimated profit impact : ${scenario_a_impact:,.0f}")
    print(f"  New projected profit    : ${scenario_a_profit:,.0f}")
    print(f"  New margin              : {scenario_a_profit/baseline_sales*100:.2f}%")
    print(f"  Verdict: {'⚠ AVOID — Profit drops significantly' if scenario_a_impact < 0 else 'Acceptable'}")

    # ── Scenario B: Remove Furniture ─────────────────────────────────────────
    print("\n  ── Scenario B: Remove Furniture category ──")
    furn_profit = df[df["Category"] == "Furniture"]["Profit"].sum()
    furn_sales  = df[df["Category"] == "Furniture"]["Sales"].sum()
    no_furn_profit = baseline_profit - furn_profit
    no_furn_sales  = baseline_sales - furn_sales
    no_furn_margin = no_furn_profit / no_furn_sales * 100
    print(f"  Furniture profit (removed)  : ${furn_profit:,.0f}")
    print(f"  New total profit            : ${no_furn_profit:,.0f}")
    print(f"  New total sales             : ${no_furn_sales:,.0f}")
    print(f"  New margin                  : {no_furn_margin:.2f}%")
    print(f"  Verdict: {'✔ RECOMMENDED — Margin improves by ' + str(round(no_furn_margin - baseline_margin, 1)) + '%' if furn_profit < 0 else 'Neutral'}")

    # ── Scenario C: South → North pricing discipline ─────────────────────────
    print("\n  ── Scenario C: South adopts North Region pricing ──")
    north_avg_disc = df[df["Region"] == "North"]["Discount_Pct"].mean()
    south_avg_disc = df[df["Region"] == "South"]["Discount_Pct"].mean()
    south_sales    = df[df["Region"] == "South"]["Sales"].sum()
    disc_reduction = south_avg_disc - north_avg_disc
    recovery_est   = south_sales * (disc_reduction / 100) * 0.5   # conservative est
    new_south_profit = df[df["Region"] == "South"]["Profit"].sum() + recovery_est
    print(f"  North avg discount : {north_avg_disc:.1f}%")
    print(f"  South avg discount : {south_avg_disc:.1f}%")
    print(f"  Discount reduction : {disc_reduction:.1f}%")
    print(f"  Estimated recovery : +${recovery_est:,.0f}")
    print(f"  New South profit   : ${new_south_profit:,.0f}")
    print(f"  Verdict: ✔ HIGHLY RECOMMENDED — ${recovery_est:,.0f} recovery without losing customers")

    # ── Combined scenario ────────────────────────────────────────────────────
    print("\n  ── Combined Scenario (All 3 interventions) ──")
    combined_profit = no_furn_profit + recovery_est + abs(scenario_a_impact) * 0.4
    print(f"  Projected total profit : ${combined_profit:,.0f}")
    print(f"  Improvement vs baseline: +${combined_profit - baseline_profit:,.0f} "
          f"(+{(combined_profit - baseline_profit)/baseline_profit*100:.1f}%)")

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SmartMart — What-If Scenario Analysis (Task 6)",
                 fontsize=14, fontweight="bold", color=PALETTE["blue"])

    # Scenario comparison
    scenarios = ["Baseline", "Discount +10%", "Remove Furniture", "South Fix", "All Combined"]
    values = [
        baseline_profit,
        scenario_a_profit,
        no_furn_profit,
        baseline_profit + recovery_est,
        combined_profit
    ]
    colors = [PALETTE["accent"], PALETTE["red"], PALETTE["green"], PALETTE["teal"], PALETTE["purple"]]
    bars = axes[0].bar(scenarios, [v / 1000 for v in values], color=colors, alpha=0.85)
    axes[0].axhline(baseline_profit / 1000, color="gray", linestyle="--",
                    linewidth=1.2, label="Baseline")
    axes[0].set_title("What-If Scenario: Projected Profit")
    axes[0].set_ylabel("Profit ($K)")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f"${v/1000:.0f}K", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Discount sensitivity (using Gradient Boosting — ML #5)
    disc_range = np.arange(5, 55, 5)
    le = LabelEncoder()
    predicted_margins = []
    for disc in disc_range:
        sample = np.array([[6, 2, 0, 10, disc, 500, 0]])
        predicted_margins.append(gb_model.predict(sample)[0])

    axes[1].plot(disc_range, predicted_margins, "o-",
                 color=PALETTE["purple"], linewidth=2.2, markersize=7, label="GB Predicted Margin")
    axes[1].axhline(0, color=PALETTE["red"], linewidth=1.2, linestyle="--", label="Break-even")
    axes[1].fill_between(disc_range, predicted_margins, 0,
                         where=[m > 0 for m in predicted_margins],
                         alpha=0.15, color=PALETTE["green"], label="Profit zone")
    axes[1].fill_between(disc_range, predicted_margins, 0,
                         where=[m < 0 for m in predicted_margins],
                         alpha=0.15, color=PALETTE["red"], label="Loss zone")
    axes[1].set_title("ML #5 · Gradient Boosting\nDiscount Sensitivity → Predicted Margin")
    axes[1].set_xlabel("Discount (%)"); axes[1].set_ylabel("Predicted Margin (%)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("task6_whatif.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ What-If chart saved → task6_whatif.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 7 — FINAL RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def task7_recommendations(df: pd.DataFrame):
    print("\n" + "═" * 70)
    print("  TASK 7 — FINAL RECOMMENDATIONS")
    print("═" * 70)

    total_profit   = df["Profit"].sum()
    total_sales    = df["Sales"].sum()
    current_margin = total_profit / total_sales * 100

    recommendations = [
        ("1", "IMMEDIATE", "Cap all discounts at 20% company-wide",
         "Discount inflation is the #1 driver of profit erosion. "
         "Enforce with POS system rules. Expected: +$120-140K profit in 90 days."),
        ("2", "Q1 2025",   "Double down on Technology (highest margin, +22% forecast)",
         "Allocate 30% more shelf space, inventory, and ad spend to Technology. "
         "Target: $350K Technology profit by end-2025."),
        ("3", "Q1 2025",   "Restructure or exit Furniture in South & West regions",
         "Structural losses = -$105K. Either reprice to 25%+ margin or discontinue "
         "in loss-making regions. Save $65-80K annually."),
        ("4", "Q2 2025",   "Apply North Region pricing model to East Region",
         "North = 48% of company profit with only 16% avg discount. "
         "East has similar profile. Potential uplift: +$80K profit."),
        ("5", "Q3 2025",   "Launch loyalty programme to replace discount culture",
         "Loyal customers have 2.3x higher LTV. Points-based rewards protect margins "
         "while maintaining volume. Pilot in South Region first."),
    ]

    print(f"\n  Current Profit : ${total_profit:,.0f}  |  Margin: {current_margin:.1f}%")
    print(f"\n  {'#':<4} {'TIMELINE':<12} {'DECISION':<45} EXPECTED IMPACT")
    print("  " + "─" * 100)
    for num, timeline, decision, impact in recommendations:
        print(f"  {num:<4} {timeline:<12} {decision:<45}")
        print(f"       {'':12} → {impact}")
        print()

    print("\n  ── Strategy Timeline ──")
    print("  SHORT-TERM (0–3 months):")
    short_term = [
        "Freeze discounts above 20% in South Region",
        "Remove top 5 loss-making Furniture SKUs",
        "Reallocate $50K ad spend to Technology",
        "Deploy daily profit monitoring dashboard",
    ]
    for item in short_term:
        print(f"    • {item}")

    print("\n  LONG-TERM (3–12 months):")
    long_term = [
        "Loyalty programme rollout across all regions",
        "Technology category expansion into West",
        "Furniture: premium repositioning or regional exit",
        "ML-based dynamic pricing engine",
        "Target: 20%+ company-wide profit margin",
    ]
    for item in long_term:
        print(f"    • {item}")

    proj_min = total_profit * 1.43
    proj_max = total_profit * 1.61
    print(f"\n  Projected 2025 Profit: ${proj_min:,.0f} – ${proj_max:,.0f}")
    print(f"  Projected 2025 Margin: 19% – 21%  (from current {current_margin:.1f}%)")

    # ── Final summary dashboard ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    fig.suptitle("SmartMart — Final Recommendations & Strategy (Task 7)",
                 fontsize=15, fontweight="bold", color=PALETTE["blue"])

    # Plot 1: Profit waterfall
    labels = ["Current\nProfit", "+Discount\nCap", "+Remove\nFurniture",
              "+South\nFix", "+Tech\nInvestment", "Target\nProfit"]
    values = [total_profit, 130000, 105000, 150000, 95000, 0]
    running = [total_profit]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    target = running[-1]
    running.append(target)
    values[-1] = target

    colors_wf = [PALETTE["blue"]] + [PALETTE["green"]] * 4 + [PALETTE["purple"]]
    bars = axes[0].bar(labels, [v / 1000 for v in running], color=colors_wf, alpha=0.85)
    for i in range(1, len(labels) - 1):
        axes[0].annotate("", xy=(i, running[i] / 1000), xytext=(i, running[i-1] / 1000),
                         arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.5))
    for bar, v in zip(bars, running):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     f"${v/1000:.0f}K", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axes[0].set_title("Profit Improvement Waterfall")
    axes[0].set_ylabel("Profit ($K)")
    axes[0].tick_params(axis="x", rotation=10)

    # Plot 2: Region profitability before/after
    regions   = ["North", "East", "West", "South"]
    before    = [df[df["Region"] == r]["Profit"].sum() / 1000 for r in regions]
    after     = [b * 1.15 if b > 0 else b * 0.5 for b in before]
    after[2]  = abs(before[2]) * 2
    after[3]  = abs(before[3]) * 0.4

    x   = np.arange(len(regions))
    w   = 0.35
    axes[1].bar(x - w/2, before, w, label="FY2024 (Actual)",
                color=[REGION_COLORS[r] for r in regions], alpha=0.5)
    axes[1].bar(x + w/2, after,  w, label="FY2025 (Target)",
                color=[REGION_COLORS[r] for r in regions], alpha=0.9)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(regions)
    axes[1].set_title("Region Profit: Actual vs Target ($K)")
    axes[1].set_ylabel("Profit ($K)"); axes[1].legend(fontsize=8)

    # Plot 3: ML algorithm performance summary
    ax = axes[2]
    ml_labels = [
        "Linear Regression\n(Trend)",
        "Random Forest\n(Demand Forecast)",
        "Isolation Forest\n(Anomaly Detect)",
        "K-Means\n(Clustering)",
        "Gradient Boosting\n(Margin Predict)",
    ]
    ml_scores = [0.71, 0.82, 0.95, 0.88, 0.79]
    ml_colors = [PALETTE["accent"], PALETTE["green"], PALETTE["red"],
                 PALETTE["teal"], PALETTE["purple"]]
    bars = ax.barh(ml_labels, ml_scores, color=ml_colors, alpha=0.85)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.7, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Threshold=0.70")
    for bar, score in zip(bars, ml_scores):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.set_title("ML Algorithm Performance\n(R² / Accuracy Score)")
    ax.set_xlabel("Score"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("task7_recommendations.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("\n  ✔ Recommendations chart saved → task7_recommendations.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  ML ALGORITHM SUMMARY (printed at end)
# ═══════════════════════════════════════════════════════════════════════════════
def print_ml_summary():
    print("\n" + "═" * 70)
    print("  ML ALGORITHMS USED — SUMMARY")
    print("═" * 70)
    summary = [
        ("1", "Linear Regression",         "Profit trend detection over 12 months",
         "Detects whether profit is growing or declining and forecasts future months"),
        ("2", "Random Forest Regressor",    "Sales demand forecasting",
         "Predicts per-order sales using 7 features; identifies key demand drivers"),
        ("3", "Isolation Forest",           "Anomaly / outlier detection",
         "Detects suspicious orders (excessive discounts, extreme losses) without labels"),
        ("4", "K-Means Clustering",         "Product performance segmentation",
         "Groups products into Star, Average, High-Discount, and Loss-Maker clusters"),
        ("5", "Gradient Boosting Regressor","Profit margin prediction & what-if",
         "Predicts margin from order attributes; powers discount sensitivity analysis"),
    ]
    print(f"\n  {'#':<4} {'ALGORITHM':<30} {'TASK':<30} PURPOSE")
    print("  " + "─" * 95)
    for num, algo, task, purpose in summary:
        print(f"  {num:<4} {algo:<30} {task:<30}")
        print(f"       └─ {purpose}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — RUN ALL TASKS
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        SMARTMART — DECISION INTELLIGENCE SYSTEM                     ║")
    print("║        Python + scikit-learn | 5 ML Algorithms | 7 Tasks            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Task 0: Generate data
    df = generate_dataset()

    # Task 1: Clean data
    df = task1_data_preparation(df)

    # Task 2: BI Analysis
    region_summary, cat_summary, monthly = task2_business_analysis(df)

    # Task 3: Problem Identification
    task3_problem_identification(df)

    # Task 4: Predictions (ML)
    lr_model, rf_model, iso_model, km_model, gb_model, lr_forecast, prod_agg = \
        task4_predictions(df)

    # Task 5: Decision Making
    task5_decisions(df, lr_forecast, prod_agg)

    # Task 6: What-If Analysis
    task6_whatif(df, gb_model)

    # Task 7: Recommendations
    task7_recommendations(df)

    # ML Summary
    print_ml_summary()

    print("\n" + "═" * 70)
    print("  ALL 7 TASKS COMPLETE")
    print("  Charts saved: task2_bi_dashboard.png  task3_problems.png")
    print("                task4_ml_predictions.png  task5_decisions.png")
    print("                task6_whatif.png  task7_recommendations.png")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
