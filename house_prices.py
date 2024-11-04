import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv('https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv')


imputer = SimpleImputer(strategy='median') 
data['total_bedrooms'] = imputer.fit_transform(data[['total_bedrooms']])

data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# separate features and target variable
X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')


model_and_scaler = {
    "model": model,
    "scaler": scaler
}



# test
example_data = {
    'longitude': -122.23,
    'latitude': 37.88,
    'housing_median_age': 41,
    'total_rooms': 5,
    'total_bedrooms': 3.0, 
    'population': 322,
    'households': 126,
    'median_income': 8.3252,
    'ocean_proximity': 'NEAR BAY'
}

example_df = pd.DataFrame([example_data])

example_df = pd.get_dummies(example_df, columns=['ocean_proximity'], drop_first=True)

# all one-hot columns
for col in X_train.columns:
    if col not in example_df.columns:
        example_df[col] = 0  # adding missing columns and fill with 0



example_scaled = scaler.transform(example_df)

# test the prediction
example_prediction = model.predict(example_scaled)

print(f"Predicted median house value for the example: {example_prediction[0]:.2f}")