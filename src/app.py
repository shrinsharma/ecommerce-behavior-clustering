# ================= STEP 14: app.py - Deployment for ecommerce clustering =================
# Streamlit web app - loads 3 pkls from models/ and predicts cluster live

import os
import pickle
import pandas as pd
import streamlit as st

# Set page title
st.set_page_config(page_title="ECommerce Behavior Clustering - Deployment", layout="wide")

# Title
st.title("🛒 ECommerce Behavior Clustering - k=4 Deployment")
st.markdown("**Full data: 1M events -> 182k sessions -> 15 features -> k=4 KMeans**")
st.markdown("Path: `[STRIPPED 82 bytes]`")

# Load models - cached
@st.cache_resource
def load_models():
    model_path = 'models/'
    if not os.path.exists(os.path.join(model_path, 'kmeans_k4_full_182k.pkl')):
        model_path = '../models/'
    with open(os.path.join(model_path, 'scaler_full_182k.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_path, 'kmeans_k4_full_182k.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(model_path, 'pca_2d_full.pkl'), 'rb') as f:
        pca = pickle.load(f)
    return scaler, kmeans, pca

scaler, kmeans, pca = load_models()
st.success(f"✅ Models loaded: scaler {scaler.n_features_in_} features, kmeans k={kmeans.n_clusters}, pca 2D")

FEATURES = ['event_count', 'unique_products', 'avg_price', 'max_price', 'min_price',
            'total_views', 'total_carts', 'total_purchases', 'duration_seconds',
            'has_purchase', 'view_to_purchase_rate', 'cart_to_purchase_rate',
            'hour_first', 'top_category_encoded', 'top_brand_encoded']

# Sidebar inputs - 15 features
st.sidebar.header("Enter New Session Behavior - 15 features")

event_count = st.sidebar.slider("event_count", 1, 50, 5)
unique_products = st.sidebar.slider("unique_products", 1, 20, 3)
avg_price = st.sidebar.slider("avg_price", 100, 10000, 1100)
max_price = st.sidebar.slider("max_price", 100, 15000, 1200)
min_price = st.sidebar.slider("min_price", 10, 10000, 800)
total_views = st.sidebar.slider("total_views", 0, 50, 3)
total_carts = st.sidebar.slider("total_carts", 0, 10, 1)
total_purchases = st.sidebar.slider("total_purchases", 0, 10, 1)
duration_seconds = st.sidebar.slider("duration_seconds", 0, 5000, 300)
has_purchase = st.sidebar.selectbox("has_purchase", [0,1], index=1)
view_to_purchase_rate = st.sidebar.slider("view_to_purchase_rate", 0.0, 1.0, 0.33)
cart_to_purchase_rate = st.sidebar.slider("cart_to_purchase_rate", 0.0, 1.0, 1.0)
hour_first = st.sidebar.slider("hour_first - first event hour", 0, 23, 10)
top_category_encoded = st.sidebar.slider("top_category_encoded", 0, 200, 50)
top_brand_encoded = st.sidebar.slider("top_brand_encoded", 0, 5000, 1000)

# Predict button
if st.sidebar.button("Predict Cluster"):
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
    input_df = pd.DataFrame([input_dict])[FEATURES]
    input_scaled = scaler.transform(input_df)
    cluster = kmeans.predict(input_scaled)[0]

    # PCA for plot
    pca_point = pca.transform(input_scaled)

    cluster_info = {
        0: {"name": "Value/Budget Conscious", "color": "blue", "offer": "Give 10% coupon + budget bundle"},
        1: {"name": "Premium High-Value", "color": "gold", "offer": "Show premium collection + free express shipping"},
        2: {"name": "Window Shoppers", "color": "orange", "offer": "Retarget with wishlist reminder + limited time deal"},
        3: {"name": "Decisive Mainstream - 71.79%", "color": "green", "offer": "Fast checkout + related products - majority segment"}
    }
    info = cluster_info[cluster]

    st.subheader(f"Predicted Cluster: {cluster} - {info['name']}")
    st.markdown(f"**Business Offer:** {info['offer']}")
    st.markdown(f"**PCA 2D point for visualization:** x={pca_point[0][0]:.2f}, y={pca_point[0][1]:.2f}")
    st.json(input_dict)

# Footer
st.markdown("---")
st.markdown("**How to run:** `streamlit run src/app.py` in terminal | **GitHub connected to VS Code**")