import json
import pickle
from pathlib import Path
from datetime import date, timedelta  
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .xgb_models import TwoStepXGBoostModel, TwoStepXGBoostWrapper, LightGBMWrapper


# ----------------------------------------------------------------------
# 1.  Load every .pkl model file once at startup
# ----------------------------------------------------------------------

MODEL_DIR = Path(r"C:\Pavan\FYP v3\websites\website v6\retail_forecaster\retail_forecaster\forecaster\models")
MODELS: dict[str, object] = {}                          # {'P001': prophet-model, ...}


def _load_models() -> None:
    """
    Load Prophet models (.pkl), XGBoost model dictionaries (.pkl), and LightGBM models (.pkl)
    """
    for file in MODEL_DIR.glob("*.pkl"):
        try:
            with open(file, "rb") as f:
                loaded_obj = pickle.load(f)
            
            # Check if it's a dictionary (XGBoost or LightGBM model components)
            if isinstance(loaded_obj, dict):
                print(f"DEBUG: {file.name} is dict with keys: {list(loaded_obj.keys())}")
                
                if loaded_obj.get('model_type') == 'TwoStepXGBoost':
                    # XGBoost 2-step model
                    wrapper = TwoStepXGBoostWrapper(loaded_obj)
                    MODELS[file.stem] = wrapper
                    print(f"[XGBOOST LOADED] {file.name}")
                    
                elif loaded_obj.get('model_type') == 'LightGBM':
                    # LightGBM model - with detailed debug
                    print(f"DEBUG: Loading LightGBM model from {file.name}")
                    wrapper = LightGBMWrapper(loaded_obj)
                    MODELS[file.stem] = wrapper
                    print(f"[LIGHTGBM LOADED] {file.name}")
                    
                elif file.stem in loaded_obj:
                    # Old incorrect format: {'P004': model_object}
                    print(f"[MODEL SKIPPED] {file.name} - old dictionary format detected")
                    print(f"  ⚠️  Please retrain {file.stem} using the updated Colab script")
                    
                else:
                    # Unknown dictionary format
                    print(f"[MODEL SKIPPED] {file.name} - unknown dictionary format")
                    print(f"  Keys found: {list(loaded_obj.keys())}")
                    
            else:
                # Assume it's a Prophet model or other compatible model
                MODELS[file.stem] = loaded_obj
                print(f"[MODEL LOADED] {file.name}")
                
        except Exception as exc:
            print(f"[MODEL ERROR] Could not load {file.name}: {exc}")



_load_models()           # run once when Django imports this module

print("-> Looking for .pkl in:", MODEL_DIR.resolve())
print("-> Models that loaded:", list(MODELS.keys()))
for product_id, model_obj in MODELS.items():
    print(f"Product {product_id}: {type(model_obj)}")
    if isinstance(model_obj, dict):
        print(f"  ⚠️  {product_id} is a dict with keys: {list(model_obj.keys())}")

# ----------------------------------------------------------------------
# 2.  Template views (no business logic)
# ----------------------------------------------------------------------

def landing_page(request):
    return render(request, "landing.html")


def dashboard_page(request):
    return render(request, "dashboard.html")


# ----------------------------------------------------------------------
# 3.  Forecast API
# ----------------------------------------------------------------------

# Mapping helpers – adjust if your training pipeline changes
PROMO_NAME_DUMMIES  = ["Diwali Dhamaka", "Monsoon Sale"]
PROMO_TYPE_DUMMIES  = ["Flat off", "Percentage Discount"]
PROMO_SCOPE_DUMMIES = ["Store-wide", "Category"]
LOCAL_EVENT_DUMMIES = ["Cricket Match Final"]
WEATHER_DUMMIES     = ["Sunny"]              # “Rainy” captured by all-zeros


def _scenario_to_features(scen: dict, week_index: int) -> list[float]:
    """
    Convert scenario to ALL features that XGBoost model expects
    """
    import datetime
    
    # Get current month for time feature
    current_month = datetime.datetime.now().month + (week_index // 4)  # Approximate month progression
    if current_month > 12:
        current_month = ((current_month - 1) % 12) + 1
    
    # Build comprehensive feature list matching training data
    features = []
    
    # Promotion Name dummies (including None)
    promo_name = scen.get("promotion_name", "None")
    for name in ["Diwali Dhamaka", "Monsoon Sale", "None"]:  # Added "None"
        features.append(1.0 if promo_name == name else 0.0)
    
    # Promotion Type dummies (including None)  
    promo_type = scen.get("promotion_type", "None")
    for ptype in ["Flat off", "Percentage Discount", "None"]:  # Added "None"
        features.append(1.0 if promo_type == ptype else 0.0)
    
    # Discount value
    features.append(float(scen.get("discount_value", 0.0)))
    
    # Promotion Scope dummies (including None)
    promo_scope = scen.get("promotion_scope", "None") 
    for scope in ["Store-wide", "Category", "None"]:  # Added "None"
        features.append(1.0 if promo_scope == scope else 0.0)
    
    # Competitor promotion
    features.append(float(scen.get("competitor_promotion_active", 0)))
    
    # Local Event dummies (including None)
    local_event = scen.get("local_event", "None")
    for event in ["Cricket Match Final", "None"]:  # Added "None"
        features.append(1.0 if local_event == event else 0.0)
    
    # Weather dummies
    weather = scen.get("weather_condition", "Sunny")
    for wthr in ["Sunny"]:  # Rainy captured by all-zeros
        features.append(1.0 if weather == wthr else 0.0)
    
    # Historical features (use defaults since we don't have actual history)
    features.append(float(scen.get("lag_1", 50.0)))          # Default previous week sales
    features.append(float(scen.get("lag_2", 48.0)))          # Default 2 weeks ago sales  
    features.append(float(scen.get("rolling_mean_4", 45.0))) # Default 4-week average
    
    # Time features
    features.append(float(current_month))                     # Current month
    
    # Unit price and stockout (if your model has them)
    features.append(float(scen.get("unit_price_inr", 0)))
    features.append(float(scen.get("stockout_flag", 0)))
    
    return features

def _scenario_to_lightgbm_features(scen: dict, week_index: int) -> list[float]:
    """
    Convert scenario to EXACT features that P004 LightGBM model expects
    Based on debug output: 16 features in exact order
    """
    from datetime import date, timedelta
    
    # Calculate target date for time features
    start_date = date.today()
    target_date = start_date + timedelta(weeks=week_index)
    
    # Build features in EXACT order as shown in debug output
    features = []
    
    # 0: Unit_Price_INR
    features.append(float(scen.get("unit_price_inr", 200.0)))  # Default price for Organic Avocados
    
    # 1: Discount_Value
    features.append(float(scen.get("discount_value", 0.0)))
    
    # 2: Competitor_Promotion_Active
    features.append(float(scen.get("competitor_promotion_active", 0)))
    
    # 3: Stockout_Flag
    features.append(float(scen.get("stockout_flag", 0)))
    
    # 4: month
    features.append(float(target_date.month))
    
    # 5: week_of_year
    features.append(float(target_date.isocalendar()[1]))  # ISO week number (1-53)
    
    # 6: day_of_week
    features.append(float(target_date.weekday()))  # Monday=0, Sunday=6
    
    # 7: lag_1 (previous week sales)
    features.append(float(scen.get("lag_1", 16.0)))  # Default historical value
    
    # 8: lag_2 (2 weeks ago sales)
    features.append(float(scen.get("lag_2", 15.0)))
    
    # 9: lag_4 (4 weeks ago sales)
    features.append(float(scen.get("lag_4", 14.0)))
    
    # 10: lag_8 (8 weeks ago sales)
    features.append(float(scen.get("lag_8", 13.0)))
    
    # 11: rolling_mean_4 (4-week rolling average)
    features.append(float(scen.get("rolling_mean_4", 15.0)))
    
    # 12: rolling_std_4 (4-week rolling standard deviation)
    features.append(float(scen.get("rolling_std_4", 2.5)))
    
    # 13: Promotion_Type_Percentage Discount (one-hot encoded)
    promo_type = scen.get("promotion_type", "None")
    features.append(1.0 if promo_type == "Percentage Discount" else 0.0)
    
    # 14: Weather_Condition_Sunny (one-hot encoded)
    weather = scen.get("weather_condition", "Sunny")
    features.append(1.0 if weather == "Sunny" else 0.0)
    
    # 15: Promotion_Scope_Store-wide (one-hot encoded)
    promo_scope = scen.get("promotion_scope", "None")
    features.append(1.0 if promo_scope == "Store-wide" else 0.0)
    
    return features




@csrf_exempt
@require_http_methods(["POST"])
def forecast_api(request):
    try:
        data = json.loads(request.body)
        product_id = data.get("product_id")
        weeks_to_forecast = int(data.get("weeks_to_forecast", 4))
        scenarios = data.get("scenarios", [])

        if product_id not in MODELS:
            return JsonResponse({"error": f"No model found for '{product_id}'"}, status=400)

        model = MODELS[product_id]
        
        # Check model type
        if hasattr(model, '__class__'):
            model_type = model.__class__.__name__
        else:
            model_type = type(model).__name__
            
        if model_type == "LightGBMWrapper":
            # Use LightGBM feature builder
            X = [_scenario_to_lightgbm_features(scen, i) for i, scen in enumerate(scenarios)]
            
            # Use EXACT feature columns from debug output
            feature_cols = [
                "Unit_Price_INR",
                "Discount_Value", 
                "Competitor_Promotion_Active",
                "Stockout_Flag",
                "month",
                "week_of_year",
                "day_of_week", 
                "lag_1",
                "lag_2",
                "lag_4",
                "lag_8",
                "rolling_mean_4",
                "rolling_std_4",
                "Promotion_Type_Percentage Discount",
                "Weather_Condition_Sunny",
                "Promotion_Scope_Store-wide"
            ]
            
            print(f"DEBUG: Creating {len(X)} rows with {len(X[0])} features each")
            print(f"DEBUG: Expected {len(feature_cols)} feature columns")
            
        else:
            # Use existing feature builder for Prophet/XGBoost
            X = [_scenario_to_features(scen, i) for i, scen in enumerate(scenarios)]
            
            # Use existing feature columns (your current Prophet/XGBoost features)
            feature_cols = [
                "Promotion_Name_Diwali Dhamaka",
                "Promotion_Name_Monsoon Sale",
                "Promotion_Name_None",
                "Promotion_Type_Flat off",
                "Promotion_Type_Percentage Discount", 
                "Promotion_Type_None",
                "Discount_Value",
                "Promotion_Scope_Store-wide",
                "Promotion_Scope_Category",
                "Promotion_Scope_None",
                "Competitor_Promotion_Active",
                "Local_Event_Cricket Match Final",
                "Local_Event_None",
                "Weather_Condition_Sunny",
                "lag_1",
                "lag_2",
                "rolling_mean_4",
                "month",
                "Unit_Price_INR",
                "Stockout_Flag",
            ]

        # Create DataFrame
        X_df = pd.DataFrame(X, columns=feature_cols)
        
        print(f"DEBUG: DataFrame shape: {X_df.shape}")

        # Add 'ds' column for Prophet models only
        if model_type not in ["LightGBMWrapper", "TwoStepXGBoostWrapper"]:
            start = date.today()
            X_df["ds"] = [start + timedelta(weeks=i) for i in range(len(X_df))]

        # Get predictions
        forecast_result = model.predict(X_df)
        predictions = forecast_result["yhat"].tolist()

        # Create response
        results = []
        start_date = date.today()
        for i, pred in enumerate(predictions):
            week_date = start_date + timedelta(weeks=i)
            results.append({
                "week": week_date.strftime("%Y-%m-%d"),
                "predicted_quantity": round(pred, 2)
            })

        return JsonResponse({
            "product_id": product_id,
            "forecasts": results
        })

    except Exception as exc:
        return JsonResponse({"error": f"Model prediction failed: {exc}"}, status=500)

