import joblib
import numpy as np

model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")

labels = encoders["perf"].classes_

def predict_employee(data):
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)[0]
    return labels[pred]