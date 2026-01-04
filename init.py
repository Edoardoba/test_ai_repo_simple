# simple_ai.py
# A very simple AI / ML example using scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Create a simple dataset
# X = numbers, y = 0 if even, 1 if odd
X = np.array([[i] for i in range(100)])
y = np.array([i % 2 for i in range(100)])

# 2. Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Make predictions
predictions = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")

# 6. Try the model on new data
new_numbers = np.array([[7], [10], [23], [42]])
results = model.predict(new_numbers)

for num, result in zip(new_numbers.flatten(), results):
    label = "odd" if result == 1 else "even"
    print(f"{num} is predicted as {label}")
