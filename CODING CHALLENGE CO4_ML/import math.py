import math
from collections import Counter

# Small training dataset
# Features: [Height, Weight]
X_train = [
    [5.0, 50],
    [5.2, 55],
    [5.5, 60],
    [5.8, 65],
    [6.0, 70],
    [6.2, 75],
    [6.5, 80],
    [6.7, 85]
]

# Class labels
y_train = [
    "A", "A", "A", "B",
    "B", "B", "B", "B"
]

# Test dataset
X_test = [
    [5.3, 57],
    [6.1, 72],
    [5.1, 52],
    [6.6, 82]
]

# Actual test labels
y_test = [
    "A", "B", "A", "B"
]


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return math.sqrt(distance)


# KNN prediction function
def knn_predict(X_train, y_train, test_point, k):

    distances = []

    # Calculate distance from test point to every training point
    for i in range(len(X_train)):
        distance = euclidean_distance(test_point, X_train[i])
        distances.append((distance, y_train[i]))

    # Sort based on distance
    distances.sort(key=lambda x: x[0])

    # Select K nearest neighbors
    nearest_neighbors = distances[:k]

    # Get their class labels
    labels = [label for distance, label in nearest_neighbors]

    # Majority voting
    prediction = Counter(labels).most_common(1)[0][0]

    return prediction


# Get K from user
k = int(input("Enter the value of K: "))

# Check valid K
if k <= 0 or k > len(X_train):
    print("Invalid value of K")
else:

    predictions = []

    print("\nPredicted Class Labels:")
    
    for i in range(len(X_test)):
        prediction = knn_predict(X_train, y_train, X_test[i], k)
        predictions.append(prediction)

        print("Test Instance", i + 1, ":", X_test[i],
              "-> Predicted Class:", prediction)

    # Calculate accuracy
    correct = 0

    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct += 1

    accuracy = (correct / len(y_test)) * 100

    print("\nActual Class Labels   :", y_test)
    print("Predicted Class Labels:", predictions)
    print("Accuracy              :", accuracy, "%")