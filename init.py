import numpy as np
import datetime
import json
import hashlib
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- [Article 12: Automatic Logging] ---
logging.basicConfig(
    filename='ai_system_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CompliantAIModel:
    def __init__(self, model_type="LogisticRegression"):
        self.model = LogisticRegression()
        self.metadata = {
            "model_type": model_type,
            "created_at": str(datetime.datetime.now()),
            "lineage": {},
            "risk_assessment": "Low" # Article 9: Simplified Risk Management
        }
        
    # --- [Article 10: Data Governance & Lineage] ---
    def prepare_data(self, data_source):
        X = np.array([[i] for i in data_source])
        y = np.array([i % 2 for i in data_source])
        
        # Lineage: Hash the data to ensure integrity
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()
        self.metadata["lineage"]["data_hash"] = data_hash
        
        # Bias check: Ensure classes are balanced
        counts = np.bincount(y)
        if abs(counts[0] - counts[1]) / len(y) > 0.1:
            logging.warning("Article 10 Warning: Data Bias detected in classes.")
            
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # --- [Article 17: Quality Management System] ---
    def train(self, X_train, y_train):
        logging.info("Starting model training (Article 12 Logging).")
        self.model.fit(X_train, y_train)
        logging.info("Training complete.")

    # --- [Article 15: Accuracy & Robustness] ---
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Stress test for robustness
        noise = np.random.normal(0, 0.1, X_test.shape)
        robust_acc = accuracy_score(y_test, self.model.predict(X_test + noise))
        
        self.metadata["metrics"] = {
            "accuracy": accuracy,
            "robustness_score": robust_acc
        }
        return accuracy

    # --- [Article 14: Human Oversight] ---
    def predict_with_oversight(self, data, human_in_the_loop=True):
        raw_preds = self.model.predict(data)
        if human_in_the_loop:
            print(f"\n[HUMAN OVERSIGHT REQUIRED] System predicts: {raw_preds}")
            confirm = input("Approve these predictions? (y/n): ")
            if confirm.lower() != 'y':
                logging.warning("Human rejected AI prediction.")
                return None
        return raw_preds

# --- EXECUTION ---

# Initialize System
ai_system = CompliantAIModel()

# 1. Data Governance (Article 10)
X_train, X_test, y_train, y_test = ai_system.prepare_data(range(100))

# 2. Training & Quality (Article 17)
ai_system.train(X_train, y_train)

# 3. Accuracy & Robustness (Article 15)
acc = ai_system.evaluate(X_test, y_test)
print(f"Validated Accuracy: {acc}")

# 4. Human Oversight (Article 14)
new_data = np.array([[7], [10]])
final_results = ai_system.predict_with_oversight(new_data)

# --- [Article 11: Technical Documentation] ---
with open("technical_documentation.json", "w") as f:
    json.dump(ai_system.metadata, f, indent=4)
print("\nArticle 11: Technical Documentation generated.")
