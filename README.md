# ECommerce Behavior Clustering - Project 1
## Data: 1M -> 995k -> 182k sessions, 15 features, k=4
## Features: event_count, unique_products, avg_price, max_price, min_price, total_views, total_carts, total_purchases, duration_seconds, has_purchase, view_to_purchase_rate, cart_to_purchase_rate, hour_first, top_category_encoded, top_brand_encoded
## Model: StandardScaler 182k x 15, KMeans k=4 full data, 3 pkls in models/
## Run: python src/train.py | python src/predict.py | streamlit run src/app.py
## Clusters: 0 Value Budget, 1 Premium, 2 Window Shoppers, 3 Decisive Mainstream 71%
## Compliance: SELECT COUNT and SELECT * without LIMIT - full data used