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
MODELS: dict[str, object] = {}


def _load_models() -> None:
    """
    Load Prophet models (.pkl), XGBoost model dictionaries (.pkl), and LightGBM models (.pkl)
    """
    for file in MODEL_DIR.glob("*.pkl"):
        try:
            with open(file, "rb") as f:
                loaded_obj = pickle.load(f)
            
            if isinstance(loaded_obj, dict):
                print(f"DEBUG: {file.name} is dict with keys: {list(loaded_obj.keys())}")
                
                if loaded_obj.get('model_type') == 'TwoStepXGBoost':
                    if 'feature_columns' not in loaded_obj and 'feature_cols' in loaded_obj:
                        loaded_obj['feature_columns'] = loaded_obj['feature_cols']
                    
                    wrapper = TwoStepXGBoostWrapper(loaded_obj)
                    MODELS[file.stem] = wrapper
                    print(f"[XGBOOST LOADED] {file.name}")
                    
                elif loaded_obj.get('model_type') == 'LightGBM':
                    wrapper = LightGBMWrapper(loaded_obj)
                    MODELS[file.stem] = wrapper
                    print(f"[LIGHTGBM LOADED] {file.name}")
                    
                elif file.stem in loaded_obj:
                    print(f"[MODEL SKIPPED] {file.name} - old dictionary format")
                    
                else:
                    print(f"[MODEL SKIPPED] {file.name} - unknown format")
                    
            else:
                MODELS[file.stem] = loaded_obj
                print(f"[MODEL LOADED] {file.name}")
                
        except Exception as exc:
            print(f"[MODEL ERROR] Could not load {file.name}: {exc}")
            import traceback
            traceback.print_exc()


_load_models()

print("-> Looking for .pkl in:", MODEL_DIR.resolve())
print("-> Models that loaded:", list(MODELS.keys()))


# ----------------------------------------------------------------------
# 2.  Template views
# ----------------------------------------------------------------------

def landing_page(request):
    return render(request, "landing.html")


def dashboard_page(request):
    return render(request, "dashboard.html")


# ----------------------------------------------------------------------
# 3.  Feature Builder Functions
# ----------------------------------------------------------------------

def _scenario_to_prophet_features(scen: dict, week_index: int) -> list[float]:
    """
    Convert scenario to features for Prophet models (P001, P002, P006)
    """
    import datetime
    
    current_month = datetime.datetime.now().month + (week_index // 4)
    if current_month > 12:
        current_month = ((current_month - 1) % 12) + 1
    
    features = []
    
    # Promotion Name dummies
    promo_name = scen.get("promotion_name", "None")
    for name in ["Diwali Dhamaka", "Monsoon Sale", "None"]:
        features.append(1.0 if promo_name == name else 0.0)
    
    # Promotion Type dummies
    promo_type = scen.get("promotion_type", "None")
    for ptype in ["Flat off", "Percentage Discount", "None"]:
        features.append(1.0 if promo_type == ptype else 0.0)
    
    # Discount value
    features.append(float(scen.get("discount_value", 0.0)))
    
    # Promotion Scope dummies
    promo_scope = scen.get("promotion_scope", "None") 
    for scope in ["Store-wide", "Category", "None"]:
        features.append(1.0 if promo_scope == scope else 0.0)
    
    # Competitor promotion
    features.append(float(scen.get("competitor_promotion_active", 0)))
    
    # Local Event dummies
    local_event = scen.get("local_event", "None")
    for event in ["Cricket Match Final", "None"]:
        features.append(1.0 if local_event == event else 0.0)
    
    # Weather dummies
    weather = scen.get("weather_condition", "Sunny")
    for wthr in ["Sunny"]:
        features.append(1.0 if weather == wthr else 0.0)
    
    # Historical features
    features.append(float(scen.get("lag_1", 50.0)))
    features.append(float(scen.get("lag_2", 48.0)))
    features.append(float(scen.get("rolling_mean_4", 45.0)))
    
    # Time features
    features.append(float(current_month))
    
    # Unit price and stockout
    features.append(float(scen.get("unit_price_inr", 0)))
    features.append(float(scen.get("stockout_flag", 0)))
    
    return features


def _scenario_to_p003_features_dynamic(scen: dict, week_index: int, model_feature_cols: list) -> list[float]:
    """
    Dynamically generate features for P003 Intermittent demand model
    Uses the same approach as P005 to ensure exact feature matching
    """
    import datetime
    
    current_month = datetime.datetime.now().month + (week_index // 4)
    if current_month > 12:
        current_month = ((current_month - 1) % 12) + 1
    
    # Pre-calculate all possible values
    value_map = {
        'Promotion_Name_Diwali Dhamaka': 1.0 if scen.get("promotion_name") == "Diwali Dhamaka" else 0.0,
        'Promotion_Name_Monsoon Sale': 1.0 if scen.get("promotion_name") == "Monsoon Sale" else 0.0,
        'Promotion_Name_None': 1.0 if scen.get("promotion_name", "None") == "None" else 0.0,
        'Promotion_Type_Flat off': 1.0 if scen.get("promotion_type") == "Flat off" else 0.0,
        'Promotion_Type_Percentage Discount': 1.0 if scen.get("promotion_type") == "Percentage Discount" else 0.0,
        'Promotion_Type_None': 1.0 if scen.get("promotion_type", "None") == "None" else 0.0,
        'Discount_Value': float(scen.get("discount_value", 0.0)),
        'Promotion_Scope_Store-wide': 1.0 if scen.get("promotion_scope") == "Store-wide" else 0.0,
        'Promotion_Scope_Category': 1.0 if scen.get("promotion_scope") == "Category" else 0.0,
        'Promotion_Scope_None': 1.0 if scen.get("promotion_scope", "None") == "None" else 0.0,
        'Competitor_Promotion_Active': float(scen.get("competitor_promotion_active", 0)),
        'Local_Event_Cricket Match Final': 1.0 if scen.get("local_event") == "Cricket Match Final" else 0.0,
        'Local_Event_None': 1.0 if scen.get("local_event", "None") == "None" else 0.0,
        'Weather_Condition_Sunny': 1.0 if scen.get("weather_condition") == "Sunny" else 0.0,
        'lag_1': float(scen.get("lag_1", 50.0)),
        'lag_2': float(scen.get("lag_2", 48.0)),
        'rolling_mean_4': float(scen.get("rolling_mean_4", 45.0)),
        'month': float(current_month),
        'Unit_Price_INR': float(scen.get("unit_price_inr", 0)),
        'Stockout_Flag': float(scen.get("stockout_flag", 0)),
    }
    
    # Generate features in exact order model expects
    features = []
    for col_name in model_feature_cols:
        if col_name in value_map:
            features.append(value_map[col_name])
        else:
            features.append(0.0)
            print(f"WARNING: P003 unknown feature '{col_name}', using 0.0")
    
    return features


def _scenario_to_lightgbm_features(scen: dict, week_index: int) -> list[float]:
    """
    Convert scenario to features for P004 LightGBM model (16 features)
    """
    from datetime import date, timedelta
    
    start_date = date.today()
    target_date = start_date + timedelta(weeks=week_index)
    
    features = []
    
    features.append(float(scen.get("unit_price_inr", 200.0)))
    features.append(float(scen.get("discount_value", 0.0)))
    features.append(float(scen.get("competitor_promotion_active", 0)))
    features.append(float(scen.get("stockout_flag", 0)))
    features.append(float(target_date.month))
    features.append(float(target_date.isocalendar()[1]))
    features.append(float(target_date.weekday()))
    features.append(float(scen.get("lag_1", 16.0)))
    features.append(float(scen.get("lag_2", 15.0)))
    features.append(float(scen.get("lag_4", 14.0)))
    features.append(float(scen.get("lag_8", 13.0)))
    features.append(float(scen.get("rolling_mean_4", 15.0)))
    features.append(float(scen.get("rolling_std_4", 2.5)))
    
    promo_type = scen.get("promotion_type", "None")
    features.append(1.0 if promo_type == "Percentage Discount" else 0.0)
    
    weather = scen.get("weather_condition", "Sunny")
    features.append(1.0 if weather == "Sunny" else 0.0)
    
    promo_scope = scen.get("promotion_scope", "None")
    features.append(1.0 if promo_scope == "Store-wide" else 0.0)
    
    return features


def _scenario_to_lumpy_features_dynamic(scen: dict, week_index: int, model_feature_cols: list) -> list[float]:
    """
    Dynamically generate features for P005 Lumpy demand model
    """
    from datetime import date, timedelta
    
    start_date = date.today()
    target_date = start_date + timedelta(weeks=week_index)
    
    promo_active = 1.0 if scen.get("promotion_type", "None") != "None" else 0.0
    event_active = 1.0 if scen.get("local_event", "None") != "None" else 0.0
    
    value_map = {
        'unit_price_inr': float(scen.get("unit_price_inr", 350.0)),
        'Unit_Price_INR': float(scen.get("unit_price_inr", 350.0)),
        'discount_value': float(scen.get("discount_value", 0.0)),
        'Discount_Value': float(scen.get("discount_value", 0.0)),
        'competitor_promotion_active': float(scen.get("competitor_promotion_active", 0)),
        'Competitor_Promotion_Active': float(scen.get("competitor_promotion_active", 0)),
        'stockout_flag': float(scen.get("stockout_flag", 0)),
        'Stockout_Flag': float(scen.get("stockout_flag", 0)),
        'month': float(target_date.month),
        'week_of_year': float(target_date.isocalendar()[1]),
        'day_of_week': float(target_date.weekday()),
        'week_of_month': float((target_date.day - 1) // 7 + 1),
        'lag_1': 11.0,
        'lag_2': 10.0,
        'lag_4': 9.0,
        'lag_8': 8.0,
        'rolling_mean_4': 10.0,
        'rolling_std_4': 3.5,
        'rolling_max_4': 15.0,
        'spike_flag_lag_1': 0.0,
        'spike_flag_lag_2': 0.0,
        'spike_flag_lag_3': 0.0,
        'spike_flag_lag_4': 0.0,
        'promo_x_lag1': promo_active * 11.0,
        'promo_x_month': promo_active * target_date.month,
        'event_x_lag1': event_active * 11.0,
        'promo_x_event': promo_active * event_active,
        'Promotion_Type_Percentage Discount': 1.0 if scen.get("promotion_type") == "Percentage Discount" else 0.0,
        'Weather_Condition_Sunny': 1.0 if scen.get("weather_condition") == "Sunny" else 0.0,
        'Promotion_Scope_Store-wide': 1.0 if scen.get("promotion_scope") == "Store-wide" else 0.0,
        'Local_Event_Cricket Match Final': 1.0 if scen.get("local_event") == "Cricket Match Final" else 0.0,
    }
    
    features = []
    for col_name in model_feature_cols:
        if col_name in value_map:
            features.append(value_map[col_name])
        else:
            features.append(0.0)
            print(f"WARNING: P005 unknown feature '{col_name}', using 0.0")
    
    return features


# ----------------------------------------------------------------------
# 4.  Forecast API
# ----------------------------------------------------------------------

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
        
        # Determine model type
        if hasattr(model, '__class__'):
            model_type = model.__class__.__name__
        else:
            model_type = type(model).__name__
        
        print(f"DEBUG: Product {product_id}, Model Type: {model_type}")
        
        # ===================================================================
        # Choose the right feature builder based on product and model type
        # ===================================================================
        
        if model_type == "LightGBMWrapper":
            # P004 - Erratic demand
            print("DEBUG: Using LightGBM feature builder for P004")
            X = [_scenario_to_lightgbm_features(scen, i) for i, scen in enumerate(scenarios)]
            feature_cols = model.feature_columns
            
        elif model_type == "TwoStepXGBoostWrapper":
            # P003 or P005 - XGBoost models
            if product_id == "P005":
                # P005 - Lumpy demand
                print("DEBUG: Using Lumpy XGBoost feature builder for P005")
                print(f"DEBUG: Model expects {len(model.feature_columns)} features")
                
                X = [_scenario_to_lumpy_features_dynamic(scen, i, model.feature_columns) 
                     for i, scen in enumerate(scenarios)]
                feature_cols = model.feature_columns
                
                if X:
                    print(f"DEBUG: First row has {len(X[0])} features")
                    
            else:
                # P003 - Intermittent demand
                print("DEBUG: Using Intermittent XGBoost feature builder for P003")
                print(f"DEBUG: Model expects {len(model.feature_columns)} features")
                
                X = [_scenario_to_p003_features_dynamic(scen, i, model.feature_columns) 
                     for i, scen in enumerate(scenarios)]
                feature_cols = model.feature_columns
                
                if X:
                    print(f"DEBUG: First row has {len(X[0])} features")
                
        else:
            # Prophet models (P001, P002, P006 - Smooth demand)
            print("DEBUG: Using Prophet feature builder")
            X = [_scenario_to_prophet_features(scen, i) for i, scen in enumerate(scenarios)]
            
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
        
        # ===================================================================
        # Create DataFrame and generate predictions
        # ===================================================================

        print(f"DEBUG: Creating DataFrame with {len(X)} rows and {len(feature_cols)} columns")
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
        import traceback
        print(f"ERROR: Model prediction failed: {exc}")
        traceback.print_exc()
        return JsonResponse({"error": f"Model prediction failed: {exc}"}, status=500)
