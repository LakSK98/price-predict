from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
from features_extraction import extract_features
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

def inverse_transform(scaled_value, min_value, max_value):
    return scaled_value * (max_value - min_value) + min_value

with open("./model/real_state_model.pkl", "rb") as file:
    loaded_clf = pickle.load(file)


@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    description = data.get("description")
    house_size = data.get("house_size")
    bedrooms = data.get("bedrooms")
    bathrooms = data.get("bathrooms")
    land_size = data.get("land_size")
    type = data.get("type")
    if description and house_size and bedrooms and bathrooms and land_size:
        try:
            extracted_features = extract_features(
                description, house_size, bedrooms, bathrooms, land_size, type
            )
            prediction = loaded_clf.predict(extracted_features)
            original_price = inverse_transform(prediction[0], 38000.0, 180000000.0)
            return jsonify({"Prediction": f'{str(original_price)} LKR'}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "No valid inputs provided"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
