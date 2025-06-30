# api/app.py

from flask import Flask, request, jsonify
from data import predict, get_metrics

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    text = data.get("text", "")
    result = predict(text)
    return jsonify({"prediction": "spam" if result == 1 else "not spam"})

@app.route("/metrics", methods=["GET"])
def metrics_route():
    return jsonify(get_metrics())

if __name__ == "__main__":
    app.run(debug=True)
