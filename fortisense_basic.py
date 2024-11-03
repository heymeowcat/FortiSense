import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


data = np.random.normal(0, 1, (100, 2))  # Normal data
anomalies = np.random.normal(5, 1, (5, 2))  # Artificial anomalies
dataset = np.vstack([data, anomalies])
df = pd.DataFrame(dataset, columns=["feature1", "feature2"])

print("Sample Dataset:")
print(df.head())

# initialize and fit Isolation Forest model

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[["feature1", "feature2"]])

# Flag and log anomalies (-1 = anomaly in Isolation Forest)
anomalies = df[df['anomaly'] == -1]

print("\nDetected Anomalies:")
print(anomalies)

anomalies.to_csv("anomalies_log.csv", index=False)
print("\nAnomalies logged to 'anomalies_log.csv'")
