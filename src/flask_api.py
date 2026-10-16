from flask import Flask, request, jsonify, render_template
import os
import pickle
import pandas as pd

app = Flask(__name__, template_folder='../templates')

model_path = 'models/'
if not os.path.exists(os.path.join(model_path, 'kmeans_k4_full_182k.pkl')):
    model_path = '../models/'

with open(os.path.join(model_path, 'scaler_full_182k.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(model_path, 'kmeans_k4_full_182k.pkl'), 'rb') as f:
    kmeans = pickle.load(f)

FEATURES = ['event_count','unique_products','avg_price','max_price','min_price','total_views','total_carts','total_purchases','duration_seconds','has_purchase','view_to_purchase_rate','cart_to_purchase_rate','hour_first','top_category_encoded','top_brand_encoded']

MEANINGS = {0:"Value Budget",1:"Premium High Value",2:"Window Shoppers",3:"Decisive Mainstream 71%"}
ACTIONS = {0:"Offer 10% coupon",1:"Offer premium collection",2:"Offer retarget reminder",3:"Offer fast checkout"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        input_dict = {feat: float(data.get(feat,0)) for feat in FEATURES}
        df = pd.DataFrame([input_dict])[FEATURES]
        cluster = int(kmeans.predict(scaler.transform(df))[0])
        return jsonify({"cluster": cluster, "meaning": MEANINGS[cluster], "action": ACTIONS[cluster]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        input_dict = {}
        for feat in FEATURES:
            input_dict[feat] = float(request.form.get(feat,0))
        df = pd.DataFrame([input_dict])[FEATURES]
        cluster = int(kmeans.predict(scaler.transform(df))[0])
        result = {"cluster": cluster, "meaning": MEANINGS[cluster], "action": ACTIONS[cluster]}
        return render_template('index.html', result=result)
    except Exception as e:
        result = {"cluster": "Error", "meaning": str(e), "action": "Check inputs"}
        return render_template('index.html', result=result)

if __name__ == '__main__':
    print("Flask HTML + API running at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)