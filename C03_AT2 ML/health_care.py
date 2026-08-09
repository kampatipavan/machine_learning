import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# 1. CREATE SAMPLE PATIENT DATA
# ============================================================

np.random.seed(42)

n = 1000

heart_rate = np.random.normal(80, 15, n)
blood_pressure = np.random.normal(120, 15, n)
spo2 = np.random.normal(97, 2, n)
temperature = np.random.normal(36.8, 0.5, n)


# ============================================================
# 2. ADD SENSOR NOISE
# ============================================================

# Wearable sensors are not perfectly accurate
heart_rate = heart_rate + np.random.normal(0, 3, n)
blood_pressure = blood_pressure + np.random.normal(0, 4, n)
spo2 = spo2 + np.random.normal(0, 0.5, n)
temperature = temperature + np.random.normal(0, 0.1, n)


# ============================================================
# 3. CREATE PATIENT CONDITION
# ============================================================

# Condition becomes abnormal when multiple vital signs
# indicate a possible health problem.

abnormal_score = (
    (heart_rate > 100).astype(int)
    + (blood_pressure > 140).astype(int)
    + (spo2 < 94).astype(int)
    + (temperature > 38).astype(int)
)

condition = np.where(abnormal_score >= 2, "Abnormal", "Normal")


# ============================================================
# 4. CONVERT CONTINUOUS VALUES INTO CATEGORIES
# ============================================================

heart_rate_state = np.where(
    heart_rate > 100, "High", "Normal"
)

blood_pressure_state = np.where(
    blood_pressure > 140, "High", "Normal"
)

spo2_state = np.where(
    spo2 < 94, "Low", "Normal"
)

temperature_state = np.where(
    temperature > 38, "High", "Normal"
)


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

data = pd.DataFrame({
    "HeartRate": heart_rate_state,
    "BloodPressure": blood_pressure_state,
    "SpO2": spo2_state,
    "Temperature": temperature_state,
    "Condition": condition
})

print("\nFIRST 10 PATIENT RECORDS")
print(data.head(10))


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["Condition"]
)


# ============================================================
# 7. CREATE BAYESIAN BELIEF NETWORK
# ============================================================

model = DiscreteBayesianNetwork([
    ("HeartRate", "Condition"),
    ("BloodPressure", "Condition"),
    ("SpO2", "Condition"),
    ("Temperature", "Condition")
])


# ============================================================
# 8. LEARN PROBABILITIES FROM TRAINING DATA
# ============================================================

model.fit(train_data)


# ============================================================
# 9. CHECK MODEL
# ============================================================

print("\nBAYESIAN NETWORK")
print(model.edges())

print("\nMODEL CHECK:")
print(model.check_model())


# ============================================================
# 10. BAYESIAN INFERENCE
# ============================================================

inference = VariableElimination(model)


# ============================================================
# 11. PREDICT TEST DATA
# ============================================================

predictions = []

for _, row in test_data.iterrows():

    evidence = {
        "HeartRate": row["HeartRate"],
        "BloodPressure": row["BloodPressure"],
        "SpO2": row["SpO2"],
        "Temperature": row["Temperature"]
    }

    result = inference.map_query(
        variables=["Condition"],
        evidence=evidence
    )

    predictions.append(result["Condition"])


# ============================================================
# 12. EVALUATE MODEL
# ============================================================

actual = test_data["Condition"]

accuracy = accuracy_score(actual, predictions)

precision = precision_score(
    actual,
    predictions,
    pos_label="Abnormal"
)

recall = recall_score(
    actual,
    predictions,
    pos_label="Abnormal"
)

f1 = f1_score(
    actual,
    predictions,
    pos_label="Abnormal"
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual,
    predictions,
    labels=["Normal", "Abnormal"]
)

print("\nCONFUSION MATRIX")
print(cm)


# ============================================================
# 14. TEST A NEW PATIENT
# ============================================================

new_patient = {
    "HeartRate": "High",
    "BloodPressure": "High",
    "SpO2": "Low",
    "Temperature": "High"
}


# Calculate probability of each condition
probability = inference.query(
    variables=["Condition"],
    evidence=new_patient
)


print("\n==============================")
print("NEW PATIENT ANALYSIS")
print("==============================")

print("Heart Rate     :", new_patient["HeartRate"])
print("Blood Pressure :", new_patient["BloodPressure"])
print("SpO2           :", new_patient["SpO2"])
print("Temperature    :", new_patient["Temperature"])


print("\nPROBABILITY OF PATIENT CONDITION")
print(probability)


# ============================================================
# 15. FINAL DECISION
# ============================================================

abnormal_probability = probability.values[
    probability.state_names["Condition"].index("Abnormal")
]

print(
    "\nProbability of Abnormal Condition:",
    round(abnormal_probability * 100, 2),
    "%"
)


if abnormal_probability >= 0.70:

    print("FINAL RESULT: CRITICAL")
    print("ALERT: Immediate medical attention recommended.")

elif abnormal_probability >= 0.30:

    print("FINAL RESULT: WARNING")
    print("Monitor the patient closely.")

else:

    print("FINAL RESULT: NORMAL")
    print("No significant anomaly detected.")