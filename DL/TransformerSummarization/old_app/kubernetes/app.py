from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Загрузка модели
model = joblib.load('iris_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['sepal_length'],
                         data['sepal_width'],
                         data['petal_length'],
                         data['petal_width']]).reshape(1, -1)

    prediction = model.predict(features).tolist()
    return jsonify({'class': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)