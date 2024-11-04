import numpy as np
import pandas as pd
import pickle

with open("model_and_scaler.pkl", "rb") as f:
    loaded_data = pickle.load(f)

model = loaded_data["model"]
scaler = loaded_data["scaler"]
X_train_columns = loaded_data["X_train_columns"]

def adversarial_attack(x_df, model, scaler, X_train_columns, epsilon=1.0):
    # one-hot encode and ensure all columns are present
    if 'ocean_proximity' in x_df.columns:
        x_df = pd.get_dummies(x_df, columns=['ocean_proximity'])
    for col in X_train_columns:
        if col not in x_df.columns:
            x_df[col] = 0
    x_df = x_df[X_train_columns]

    x_scaled = scaler.transform(x_df.values.astype(float))
    y_pred = model.predict(x_scaled)

    # create a stronger perturbation based on epsilon and original prediction
    gradient = np.sign(x_scaled) * epsilon * 0.1  # Scaled perturbation

    # Apply the perturbation
    x_adv_scaled = x_scaled + gradient
    x_adv = scaler.inverse_transform(x_adv_scaled)

    return x_adv.flatten()

# Example
example_data = {
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41,
    'total_rooms': 880,
    'total_bedrooms': 129.0,
    'population': 322,
    'households': 126,
    'median_income': 8.3252,
    'ocean_proximity': 'NEAR BAY'
}

# Convert example data to DataFrame and prepare it for attack
example_df = pd.DataFrame([example_data])
example_df = pd.get_dummies(example_df, columns=['ocean_proximity'])


for col in X_train_columns:
    if col not in example_df.columns:
        example_df[col] = 0
example_df = example_df[X_train_columns]

x_adv = adversarial_attack(example_df.copy(), model, scaler, X_train_columns, epsilon=1.0)

x_adv_df = pd.DataFrame([x_adv], columns=X_train_columns)

x_adv_scaled = scaler.transform(x_adv_df)
adv_prediction = model.predict(x_adv_scaled)

original_scaled = scaler.transform(example_df)
original_prediction = model.predict(original_scaled)

print("Original Prediction:", original_prediction)
print("Adversarial Prediction:", adv_prediction)
diff = np.abs(original_prediction - adv_prediction)[0]
print("Difference after attack:", diff)
