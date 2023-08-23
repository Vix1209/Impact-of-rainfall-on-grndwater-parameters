import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Groundwater and rainfall data
groundwater_data = {
    "Year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "Temperature": [22.55, 21.5, 26.67, 29.3, 29.87, 28, 28.67, 28.97, 27.47, 27.2],
    "ph": [5.01, 5.52, 5.65, 5.3, 5.68, 5.12, 5.13, 5.67, 5.18, 6.55],
    "Total dissolved solids": [70.7, 93.43, 157.23, 135.07, 156.23, 80.17, 37.93, 58.53, 80.6, 222.8],
    "Electrical conductivity": [139.9, 183, 168.4, 165.67, 565, 180.63, 86.23, 58.67, 179.8, 497.83],
    "Iron": [0.12, 0, 1.6, 0.34, 2.57, 0.27, 0.05, 0.11, 0.01, 0.02],
    "Nitrate": [8.43, 6.67, 0.43, 1.23, 1.13, 0.07, 0, 2.1, 0.07, 0.3],
    "Magnesium": [6, 4, 9, 9.2, 2.13, 6.67, 5.2, 3.6, 0.06, 0.04]
}

rainfall_data = {
    "Year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "Rainfall": [208.43, 211.51, 182.04, 198.84, 227.74, 176.21, 203, 215.03, 188.92, 171.41]
}

# Combine the groundwater and rainfall data
combined_data = pd.merge(pd.DataFrame(groundwater_data), pd.DataFrame(rainfall_data), on='Year')

# Split data into training (2012-2018) and testing (2019-2021) sets
train_data = combined_data[combined_data['Year'] <= 2018]
test_data = combined_data[combined_data['Year'] >= 2019]

# Separate dependent variables (groundwater parameters) and independent variable (Rainfall) for training and testing
y_train = train_data.drop(columns=['Year', 'Rainfall'])
X_train = train_data['Rainfall']

y_test = test_data.drop(columns=['Year', 'Rainfall'])
X_test = test_data['Rainfall']

# Initialize and train the model
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)

# Predict the groundwater parameters for the test data
y_pred = model.predict(X_test.values.reshape(-1, 1))

# Evaluate the model (MSE and R-squared)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set MSE: {mse:.2f}, Test Set R-squared: {r2:.2f}")

# Calculate coefficient and intercept
coefficient = model.coef_  # Access the coefficient value using indexing
intercept = model.intercept_

print(f"Coefficient (Slope): {coefficient}")
print(f"Intercept: {intercept}")