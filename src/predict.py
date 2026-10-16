# ================= STEP 13 FIXED: predict.py - 15 features exact match to scaler pkl =================
import os
import pickle
import numpy as np
import pandas as pd

# Models path
model_path = 'models/'
if not os.path.exists(os.path.join(model_path, 'kmeans_k4_full_182k.pkl')):
    model_path = '../models/'

print(f"STEP 13 CHECK: Loading pkls from {model_path}")

# Load scaler - fitted on 15 features full 182k
with open(os.path.join(model_path, 'scaler_full_182k.pkl'), 'rb') as f:
    scaler = pickle.load(f)
print(f"STEP 13 CHECK: scaler loaded - expects {scaler.n_features_in_} features")

# Load kmeans k=4
with open(os.path.join(model_path, 'kmeans_k4_full_182k.pkl'), 'rb') as f:
    kmeans = pickle.load(f)
print(f"STEP 13 CHECK: kmeans loaded - k={kmeans.n_clusters}")

# Exact feature order from scaler.feature_names_in_
FEATURES = ['event_count', 'unique_products', 'avg_price', 'max_price', 'min_price',
            'total_views', 'total_carts', 'total_purchases', 'duration_seconds',
            'has_purchase', 'view_to_purchase_rate', 'cart_to_purchase_rate',
            'hour_first', 'top_category_encoded', 'top_brand_encoded']

def predict_cluster(event_count, unique_products, avg_price, max_price, min_price,
                    total_views, total_carts, total_purchases, duration_seconds,
                    has_purchase, view_to_purchase_rate, cart_to_purchase_rate,
                    hour_first, top_category_encoded, top_brand_encoded):
    # Create DataFrame with exact column names to match scaler
    input_dict = {
        'event_count': event_count,
        'unique_products': unique_products,
        'avg_price': avg_price,
        'max_price': max_price,
        'min_price': min_price,
        'total_views': total_views,
        'total_carts': total_carts,
        'total_purchases': total_purchases,
        'duration_seconds': duration_seconds,
        'has_purchase': has_purchase,
        'view_to_purchase_rate': view_to_purchase_rate,
        'cart_to_purchase_rate': cart_to_purchase_rate,
        'hour_first': hour_first,
        'top_category_encoded': top_category_encoded,
        'top_brand_encoded': top_brand_encoded
    }
    input_df = pd.DataFrame([input_dict])
    # Ensure order exactly as FEATURES
    input_df = input_df[FEATURES]
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict
    cluster = kmeans.predict(input_scaled)[0]
    cluster_meaning = {
        0: "Value/Budget Conscious - low avg_price",
        1: "Premium High-Value Shoppers",
        2: "Window Shoppers/Browsers - long duration",
        3: "Decisive Mainstream Buyers - 71.79% majority"
    }
    return cluster, cluster_meaning.get(cluster, "Unknown")

print("\nSTEP 13 CHECK: Demo predictions with 15 features:")

# Example 1: Decisive buyer - 3 events, 2 products, quick
c1, m1 = predict_cluster(3, 2, 1100, 1200, 800, 2, 1, 1, 120, 1, 0.5, 1.0, 10, 50, 1000)
print(f" - Example 1 Decisive (3 events, 1100 avg) -> Cluster {c1}: {m1}")

# Example 2: Premium - high price
c2, m2 = predict_cluster(8, 5, 4500, 6000, 3000, 4, 3, 2, 600, 1, 0.5, 0.66, 14, 20, 500)
print(f" - Example 2 Premium (4500 avg, 6000 max) -> Cluster {c2}: {m2}")

# Example 3: Window shopper - long duration no purchase
c3, m3 = predict_cluster(15, 10, 900, 1500, 200, 15, 0, 0, 1800, 0, 0.0, 0.0, 9, 100, 2000)
print(f" - Example 3 Window (15 events, 1800 sec, 0 purchase) -> Cluster {c3}: {m3}")

print("\nSTEP 13 COMPLETE: predict.py fixed with 15 features")