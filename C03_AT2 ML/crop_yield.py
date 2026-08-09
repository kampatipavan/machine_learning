import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CREATE SYNTHETIC AGRICULTURAL DATA
# ============================================================

np.random.seed(42)

n = 1000

rainfall = np.random.normal(800, 200, n)
temperature = np.random.normal(28, 4, n)
soil_quality = np.random.normal(60, 15, n)

# Seasonal information
season = np.random.choice(
    ["Summer", "Monsoon", "Winter"],
    n
)


# ============================================================
# 2. CREATE CROP YIELD
# ============================================================

yield_score = (
    (rainfall >= 600).astype(int)
    + (rainfall <= 1000).astype(int)
    + (temperature >= 20).astype(int)
    + (temperature <= 32).astype(int)
    + (soil_quality >= 60).astype(int)
)


# Seasonal influence
yield_score += np.where(
    season == "Monsoon",
    1,
    0
)


yield_category = np.where(
    yield_score >= 5,
    "High",
    np.where(
        yield_score >= 3,
        "Medium",
        "Low"
    )
)


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

data = pd.DataFrame({
    "Rainfall": rainfall,
    "Temperature": temperature,
    "SoilQuality": soil_quality,
    "Season": season,
    "Yield": yield_category
})


print("\n======================================")
print("FIRST 10 AGRICULTURAL RECORDS")
print("======================================")

print(data.head(10))


# ============================================================
# 4. CONVERT SEASON INTO NUMERIC VALUES
# ============================================================

season_mapping = {
    "Summer": 0,
    "Monsoon": 1,
    "Winter": 2
}

data["Season"] = data["Season"].map(
    season_mapping
)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = data[
    [
        "Rainfall",
        "Temperature",
        "SoilQuality",
        "Season"
    ]
]

y = data["Yield"]


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. BAYESIAN MODEL
# ============================================================

model = GaussianNB()

model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 9. PERFORMANCE EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)


print("\n======================================")
print("BAYESIAN MODEL PERFORMANCE")
print("======================================")

print(
    "Accuracy  :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score  :",
    round(f1 * 100, 2),
    "%"
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"]
)


print("\nCONFUSION MATRIX")
print(cm)


# ============================================================
# 11. BAYESIAN PRIOR PROBABILITIES
# ============================================================

print("\n======================================")
print("PRIOR PROBABILITIES")
print("======================================")


prior_probabilities = (
    y_train.value_counts(
        normalize=True
    )
)


print(
    prior_probabilities
)


# ============================================================
# 12. TEST A NEW FARM
# ============================================================

new_farm = pd.DataFrame({
    "Rainfall": [850],
    "Temperature": [27],
    "SoilQuality": [75],
    "Season": [1]
})


prediction = model.predict(
    new_farm
)

probability = model.predict_proba(
    new_farm
)


print("\n======================================")
print("NEW FARM PREDICTION")
print("======================================")


print("Rainfall    : 850 mm")
print("Temperature : 27 °C")
print("Soil Quality: 75")
print("Season      : Monsoon")


print(
    "\nPredicted Crop Yield:",
    prediction[0]
)


# ============================================================
# 13. SHOW PROBABILITIES
# ============================================================

print("\nYield Probabilities:")


for class_name, prob in zip(
    model.classes_,
    probability[0]
):

    print(
        class_name,
        ":",
        round(prob * 100, 2),
        "%"
    )


print("\n======================================")
print("PROGRAM COMPLETED")
print("======================================")