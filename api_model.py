from flask import Flask,request,jsonify
import hpelm
import numpy as np
import pickle
import pandas as pd

# Load model dan scaler
model = hpelm.ELM(16, 2)  # Sesuaikan dimensi input dan output
model.load('diabetes_model.elm')
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.json

        # Pastikan semua fitur yang dibutuhkan ada
        required_features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
                             'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
                             'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'PhysHlth', 'DiffWalk', 
                             'Sex', 'Age']

        if not all(feature in data for feature in required_features):
            return jsonify({"error": "All required features must be provided"}), 400

        # Ambil fitur dari data
        features = np.array([[data[feature] for feature in required_features]])

        # Convert ke DataFrame dengan nama kolom yang sesuai
        features_df = pd.DataFrame(features, columns=required_features)

        # Preprocessing dengan scaler
        scaled_features = scaler.transform(features_df)

        # Prediksi
        pred_prob = model.predict(scaled_features)
        pred_class = int(np.argmax(pred_prob, axis=1)[0])  # 0 atau 1

        return jsonify({"prediction": pred_class}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
