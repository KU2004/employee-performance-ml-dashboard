import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(22, 60, n),
        "Experience": np.random.randint(1, 35, n),
        "Salary": np.random.randint(20000, 150000, n),
        "Training_Hours": np.random.randint(0, 100, n),
        "Attendance": np.random.randint(50, 100, n),
        "Projects_Completed": np.random.randint(1, 20, n),
        "Department": np.random.choice(["HR", "IT", "Sales", "Finance"], n)
    })

    # Smart performance formula
    score = (
        0.4 * data["Experience"] +
        0.2 * data["Training_Hours"] +
        0.2 * data["Attendance"] +
        0.2 * data["Projects_Completed"]
    )

    data["Performance_Label"] = pd.cut(
        score,
        bins=[0, 40, 75, 150],
        labels=["Low", "Medium", "High"]
    )

    return data