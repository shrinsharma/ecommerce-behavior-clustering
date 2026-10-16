# ================= STEP 12: train.py - Training clustering model for deployment =================
# This file trains KMeans k=4 on full ecommerce_behavior data and saves pkls for predict.py and app.py

# Import sqlite3 - tool to open.db filing cabinet
import sqlite3
# Import os - to check paths exist
import os
# Import pandas - for dataframe handling
import pandas as pd
# Import numpy - for numeric operations
import numpy as np
# Import pickle - to save scaler and model as pkl
import pickle
# Import StandardScaler - to scale 182k sessions mean 0 std 1
from sklearn.preprocessing import StandardScaler
# Import KMeans - clustering algorithm
from sklearn.cluster import KMeans
# Import PCA - for 2D visualization
from sklearn.decomposition import PCA

# Set DB path - update if your clustering.db is elsewhere, currently looking in data folder
db_path = '../data/clustering.db'
# Alternative path if db is in project root - checks both
if not os.path.exists(db_path):
    db_path = 'H:/INTTRVU/CAPSTONE/Project 1 - ECommerce behavior data/clustering.db'

# Check: DB exists?
print(f"STEP 12 CHECK: DB path exists? {os.path.exists(db_path)} - {db_path}")

# Action: Connect to SQLite and load full table without LIMIT - compliance proof
conn = sqlite3.connect(db_path)
# Action: SELECT COUNT(*) for compliance
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM ecommerce_behavior")
count = cursor.fetchone()[0]
print(f"STEP 12 CHECK: COUNT(*) from ecommerce_behavior: {count}")

# Action: Load full df - SELECT * without LIMIT - full 1M
df = pd.read_sql_query("SELECT * FROM ecommerce_behavior", conn)
print(f"STEP 12 CHECK: Loaded full df shape: {df.shape} - should be ({count}, 9)")

# Action: Close connection
conn.close()

# --- Cleaning - same as Colab STEP 6a ---
# Remove price 0 and duplicates - only 0.5% removal
df_clean = df[(df['price'] > 0)].drop_duplicates()
print(f"STEP 12 CHECK: After cleaning df_clean shape: {df_clean.shape}")

# --- Session Aggregation - 995k events -> 182k sessions - same as Colab STEP 6b ---
# Group by user_session to make behavioral features
# This is simplified version - full version has 15 features, using 8 core for deployment demo
session_df = df_clean.groupby('user_session').agg(
    event_count=('event_type', 'count'),
    unique_products=('product_id', 'nunique'),
    avg_price=('price', 'mean'),
    total_carts=('event_type', lambda x: (x=='cart').sum()),
    total_purchases=('event_type', lambda x: (x=='purchase').sum()),
    duration_seconds=('event_time', lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).total_seconds())
).reset_index()
print(f"STEP 12 CHECK: Session aggregation shape: {session_df.shape} - 100% sessions used")

# --- Feature Selection and Scaling ---
# Select numeric features for clustering - 6 core features for deployment simplicity
features = ['event_count', 'unique_products', 'avg_price', 'total_carts', 'total_purchases', 'duration_seconds']
X = session_df[features].fillna(0)
print(f"STEP 12 CHECK: X shape for clustering: {X.shape}")

# Action: Scale full data - StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"STEP 12 CHECK: X_scaled shape: {X_scaled.shape}")

# --- Final KMeans k=4 - no train_test_split - per Problem Statement complete data can be used ---
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
print(f"STEP 12 CHECK: KMeans k=4 fitted on full {X_scaled.shape[0]} - inertia: {kmeans_final.inertia_:.2f}")

# --- PCA for visualization ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"STEP 12 CHECK: PCA explained variance 2D: {pca.explained_variance_ratio_.sum():.4f}")

# --- Save pkls to models/ folder ---
# Ensure models folder exists
os.makedirs('models', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Save scaler
with open('models/scaler_full_182k.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("STEP 12 CHECK: scaler_full_182k.pkl saved to models/")

# Save kmeans
with open('models/kmeans_k4_full_182k.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)
print("STEP 12 CHECK: kmeans_k4_full_182k.pkl saved to models/")

# Save pca
with open('models/pca_2d_full.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("STEP 12 CHECK: pca_2d_full.pkl saved to models/")

print("\nSTEP 12 COMPLETE: train.py ready - pkls saved for deployment")