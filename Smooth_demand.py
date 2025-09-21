# Training and Testing Prohet model for P001, P002, P006 


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from prophet import Prophet
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging

# Suppress verbose logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- 1. Load the Ideal Dataset ---
try:
    df = pd.read_csv('dataset.csv')
    df['Week'] = pd.to_datetime(df['Week'],dayfirst=True)
    print("Ideal dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'dataset.csv' not found. Please ensure you have uploaded the ideal dataset.")
    exit()

# --- 2. Evaluation Metrics ---
def calculate_metrics(y_true, y_pred):
    """Calculates a dictionary of evaluation metrics."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask_s = denominator != 0
    smape = np.mean(2 * np.abs(y_pred[mask_s] - y_true[mask_s]) / denominator[mask_s]) * 100 if np.any(mask_s) else 0

    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mape,
        'sMAPE': smape,
        'R-squared': r2_score(y_true, y_pred)
    }

# --- 3. Model for Smooth Demand (Prophet) ---
def run_prophet_forecast(product_df):
    prophet_df = product_df.rename(columns={'Week': 'ds', 'Quantity_Sold': 'y'})
    prophet_df = pd.get_dummies(prophet_df, columns=['Promotion_Type', 'Weather_Condition', 'Promotion_Scope', 'Local_Event'], drop_first=True)

    regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y', 'Product_ID', 'Product_Name', 'Product_Category', 'Demand_Type', 'Promotion_Name']]

    train_df = prophet_df[prophet_df['ds'].dt.year <= 2024]
    test_df = prophet_df[prophet_df['ds'].dt.year == 2025]

    model = Prophet(seasonality_mode='multiplicative')
    for reg in regressor_cols:
        model.add_regressor(reg)

    model.fit(train_df)
    future = test_df.drop(columns=['y'])



    forecast = model.predict(future)

    return test_df['y'].values, forecast['yhat'].values, test_df['ds']


# --- 6. Main Execution Loop ---
all_metrics = {}
all_results_dfs = {}
# Filter for only Smooth demand products
smooth_products = df[df['Demand_Type'] == 'Smooth']['Product_ID'].unique()

for product_id in smooth_products:
    product_df = df[df['Product_ID'] == product_id].copy()

    print(f"\n----- Processing {product_id} (Smooth) with Prophet -----")
    actuals, predictions, dates = run_prophet_forecast(product_df)

    metrics = calculate_metrics(actuals, predictions)
    all_metrics[product_id] = metrics
    all_results_dfs[product_id] = pd.DataFrame({'Actual': actuals, 'Forecast': predictions}, index=dates)

# --- 7. Display Results ---
print("\n\n--- Final Multi-Model Evaluation Metrics (2025 Forecast) ---")
metrics_summary = pd.DataFrame(all_metrics).T.sort_index()
print(metrics_summary.to_string())

# --- 8. Plotting ---
print("\n--- Generating Final Forecast Plots ---")
fig, axes = plt.subplots(nrows=len(smooth_products), ncols=1, figsize=(15, 5 * len(smooth_products)), sharex=True)
if len(smooth_products) == 1: # Handle case of single plot
    axes = [axes]

for i, product_id in enumerate(sorted(smooth_products)):
    if product_id in all_results_dfs:
        results_df = all_results_dfs[product_id]
        demand_type = df[df['Product_ID'] == product_id]['Demand_Type'].iloc[0]
        ax = axes[i]
        results_df['Actual'].plot(ax=ax, label='Actual Sales', style='-o', color='blue', alpha=0.7)
        results_df['Forecast'].plot(ax=ax, label='Forecasted Sales', style='--o', color='red', alpha=0.7)
        ax.set_title(f'Forecast vs. Actual for {product_id} (Demand Type: {demand_type})', fontsize=16)
        ax.set_ylabel('Quantity Sold')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

fig.text(0.5, 0.01, 'Week of 2025', ha='center', va='center', fontsize=12)
plt.tight_layout(rect=[0, 0.02, 1, 0.99])
plt.show()

print("\nProcess finished.")



# converting the models to a pickle file (.pkl)

# ------------ 0. Install / import essentials ------------
import logging
import pandas as pd
import numpy as np
from prophet import Prophet            # pip install prophet
import pickle                          # ← use pickle instead of joblib

# quiet Stan / Prophet logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

print("Libraries imported successfully.")

# ------------ 1. Load the ideal dataset ------------
try:
    df = pd.read_csv("dataset.csv")
    df["Week"] = pd.to_datetime(df["Week"], dayfirst=True)
    print("Ideal dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'dataset.csv' not found. Please upload the file.")
    raise SystemExit

# ------------ 2. Training-and-save routine ------------
def train_and_save_prophet_pkl(product_df: pd.DataFrame, product_id):
    """
    Trains a Prophet model and saves it to prophet_model_<Product_ID>.pkl
    (identical logic to the joblib version).
    """
    print(f"\n----- Training final Prophet model for {product_id} -----")

    # Prophet-ready frame
    prophet_df = product_df.rename(columns={"Week": "ds", "Quantity_Sold": "y"})

    # One-hot encode categorical regressors
    prophet_df = pd.get_dummies(
        prophet_df,
        columns=["Promotion_Type", "Weather_Condition", "Promotion_Scope", "Local_Event"],
        drop_first=True,
    )

    # Regressor column list
    regressor_cols = [
        c
        for c in prophet_df.columns
        if c
        not in ["ds", "y", "Product_ID", "Product_Name", "Product_Category", "Demand_Type", "Promotion_Name"]
    ]

    # Build & fit model
    model = Prophet(seasonality_mode="multiplicative")
    for reg in regressor_cols:
        model.add_regressor(reg)
    model.fit(prophet_df)
    print(f"Prophet model for {product_id} trained successfully.")

    # -------- save as .pkl --------
    file_name = f"prophet_model_{product_id}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Model saved as '{file_name}'")

# ------------ 3. Loop over smooth-demand SKUs ------------
smooth_ids = df[df["Demand_Type"] == "Smooth"]["Product_ID"].unique()

for pid in smooth_ids:
    train_and_save_prophet_pkl(df[df["Product_ID"] == pid].copy(), pid)

print("\n--- Process Complete ---")
print("All final .pkl model files for smooth-demand products have been generated.")
