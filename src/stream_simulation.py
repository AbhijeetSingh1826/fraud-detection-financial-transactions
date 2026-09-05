import pandas as pd
import joblib
import requests
import time
import json

# Load test data (we'll pretend these are transactions arriving live)
X_test = joblib.load('../models/X_test.pkl')
y_test = joblib.load('../models/y_test.pkl')

API_URL = "http://127.0.0.1:8000/predict"

def simulate_stream(n_transactions=50, delay=0.5):
    """
    Streams n_transactions from the test set to the API,
    one at a time, with a delay to mimic real-time arrival.
    """
    results = []

    # Grab a mixed sample so we see both legit and fraud cases
        # Deliberately guarantee a mix of fraud and legit cases for a meaningful demo
    fraud_indices = y_test[y_test == 1].index
    legit_indices = y_test[y_test == 0].index

    n_fraud = min(10, len(fraud_indices))  # include up to 10 real fraud cases
    n_legit = n_transactions - n_fraud

    sample_indices = list(fraud_indices[:n_fraud]) + list(pd.Series(legit_indices).sample(n=n_legit, random_state=42))
    sample = X_test.loc[sample_indices].sample(frac=1, random_state=42)  # shuffle order
    actual_labels = y_test.loc[sample.index]

    for idx, (i, row) in enumerate(sample.iterrows()):
        payload = row.to_dict()

        response = requests.post(API_URL, json=payload)
        result = response.json()

        actual = int(actual_labels.loc[i])
        predicted = result['fraud_prediction']

        log_entry = {
            "transaction_id": i,
            "actual_fraud": actual,
            "predicted_fraud": predicted,
            "fraud_probability": result['fraud_probability'],
            "risk_level": result['risk_level'],
            "correct": actual == predicted
        }
        results.append(log_entry)

        status = "🚨 FLAGGED" if predicted == 1 else "✅ Cleared"
        match = "✓" if actual == predicted else "✗ (missed)"
        print(f"Txn {idx+1}/{n_transactions} | {status} | Risk: {result['risk_level']} | "
              f"Prob: {result['fraud_probability']:.3f} | Actual: {actual} {match}")

        time.sleep(delay)  # simulates transactions arriving over time

    return pd.DataFrame(results)

if __name__ == "__main__":
    results_df = simulate_stream(n_transactions=50, delay=0.5)
    results_df.to_csv("../data/stream_results.csv", index=False)
    print("\n--- Stream Summary ---")
    print(f"Total transactions: {len(results_df)}")
    print(f"Flagged as fraud: {results_df['predicted_fraud'].sum()}")
    print(f"Correct predictions: {results_df['correct'].sum()}/{len(results_df)}")