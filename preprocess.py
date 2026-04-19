import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    le_dept = LabelEncoder()
    le_perf = LabelEncoder()

    data["Department"] = le_dept.fit_transform(data["Department"])
    data["Performance_Label"] = le_perf.fit_transform(data["Performance_Label"])

    # Save encoders
    joblib.dump({
        "dept": le_dept,
        "perf": le_perf
    }, "models/encoders.pkl")

    return data