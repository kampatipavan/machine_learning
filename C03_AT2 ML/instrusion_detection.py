import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CREATE SYNTHETIC NETWORK DATASET
# ============================================================

np.random.seed(42)

n = 1000

# Network features
packet_rate = np.random.normal(50, 20, n)
connection_duration = np.random.normal(100, 40, n)
failed_logins = np.random.poisson(2, n)
source_bytes = np.random.normal(5000, 2000, n)
destination_bytes = np.random.normal(5000, 2000, n)


# ============================================================
# 2. CREATE NORMAL / ATTACK LABEL
# ============================================================

attack_score = (
    (packet_rate > 80).astype(int)
    + (connection_duration < 50).astype(int)
    + (failed_logins > 5).astype(int)
    + (source_bytes > 8000).astype(int)
)

network_state = np.where(
    attack_score >= 2,
    "Attack",
    "Normal"
)


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

data = pd.DataFrame({
    "PacketRate": packet_rate,
    "ConnectionDuration": connection_duration,
    "FailedLogins": failed_logins,
    "SourceBytes": source_bytes,
    "DestinationBytes": destination_bytes,
    "NetworkState": network_state
})


print("\n===================================")
print("FIRST 10 NETWORK RECORDS")
print("===================================")

print(data.head(10))


# ============================================================
# 4. CREATE INCOMPLETE DATA
# ============================================================

# Simulate missing sensor/network information

data.loc[10:20, "PacketRate"] = np.nan
data.loc[30:40, "FailedLogins"] = np.nan
data.loc[50:60, "SourceBytes"] = np.nan

print("\n===================================")
print("MISSING DATA")
print("===================================")

print(data.isnull().sum())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("NetworkState", axis=1)
y = data["NetworkState"]


# ============================================================
# 6. HANDLE INCOMPLETE DATA
# ============================================================

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)


# ============================================================
# 7. SPLIT TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. NAÏVE BAYES CLASSIFIER
# ============================================================

naive_bayes = GaussianNB()

naive_bayes.fit(
    X_train,
    y_train
)


# ============================================================
# 9. PREDICT NETWORK TRAFFIC
# ============================================================

predictions = naive_bayes.predict(X_test)


# ============================================================
# 10. PERFORMANCE EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label="Attack"
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="Attack"
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="Attack"
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Normal", "Attack"]
)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]


false_positive_rate = FP / (FP + TN)


print("\n===================================")
print("NAÏVE BAYES PERFORMANCE")
print("===================================")

print("Accuracy          :", round(accuracy * 100, 2), "%")
print("Precision         :", round(precision * 100, 2), "%")
print("Recall            :", round(recall * 100, 2), "%")
print("F1 Score          :", round(f1 * 100, 2), "%")
print("False Positive Rate:", round(false_positive_rate * 100, 2), "%")


print("\nCONFUSION MATRIX")
print(cm)


# ============================================================
# 12. EXPECTATION-MAXIMIZATION (EM)
# ============================================================

print("\n===================================")
print("EXPECTATION-MAXIMIZATION")
print("===================================")


# EM does not require class labels.
# It tries to discover hidden groups in network behavior.

gmm = GaussianMixture(
    n_components=2,
    random_state=42
)

gmm.fit(X_imputed)


# Predict hidden clusters
clusters = gmm.predict(X_imputed)

data["HiddenCluster"] = clusters


print("\nHidden clusters discovered by EM:")

print(
    data["HiddenCluster"]
    .value_counts()
)


# ============================================================
# 13. ANALYZE EM CLUSTERS
# ============================================================

cluster_analysis = pd.crosstab(
    data["HiddenCluster"],
    data["NetworkState"]
)

print("\n===================================")
print("EM CLUSTER ANALYSIS")
print("===================================")

print(cluster_analysis)


# ============================================================
# 14. BAYESIAN LEARNING
# ============================================================

print("\n===================================")
print("BAYESIAN LEARNING")
print("===================================")


# Prior belief
prior_attack = 0.10

print(
    "Initial probability of attack:",
    prior_attack
)


# New observed network events
new_events = [
    "Attack",
    "Normal",
    "Attack",
    "Attack",
    "Normal"
]


# Beta prior:
# alpha = attack evidence
# beta = normal evidence

alpha = 1
beta = 9


for event in new_events:

    if event == "Attack":
        alpha += 1
    else:
        beta += 1

    posterior_probability = alpha / (alpha + beta)

    print(
        "Observed:",
        event,
        "| Updated attack probability:",
        round(posterior_probability, 3)
    )


# ============================================================
# 15. TEST A NEW NETWORK CONNECTION
# ============================================================

new_connection = np.array([[
    120,     # Packet Rate
    30,      # Connection Duration
    8,       # Failed Logins
    10000,   # Source Bytes
    4000     # Destination Bytes
]])


# Make sure preprocessing is identical
new_connection = imputer.transform(
    new_connection
)


# Prediction
prediction = naive_bayes.predict(
    new_connection
)

probability = naive_bayes.predict_proba(
    new_connection
)


# Find Attack probability
attack_index = list(
    naive_bayes.classes_
).index("Attack")

attack_probability = probability[0][attack_index]


print("\n===================================")
print("NEW NETWORK CONNECTION")
print("===================================")

print("Packet Rate        : 120")
print("Connection Duration: 30")
print("Failed Logins      : 8")
print("Source Bytes       : 10000")
print("Destination Bytes  : 4000")


print(
    "\nProbability of Attack:",
    round(attack_probability * 100, 2),
    "%"
)


if attack_probability >= 0.70:

    print("\nFINAL RESULT: ATTACK DETECTED")
    print("ALERT: Suspicious network activity!")

elif attack_probability >= 0.40:

    print("\nFINAL RESULT: SUSPICIOUS")
    print("WARNING: Further investigation required.")

else:

    print("\nFINAL RESULT: NORMAL")
    print("No significant threat detected.")


print("\n===================================")
print("PROGRAM COMPLETED")
print("===================================")