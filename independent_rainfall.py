import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    "Magnesium": [6, 4, 9, 9.2, 2.13, 6.67, 5.2, 3.6, 0.06, 0.04],
}

rainfall_data = {
    "Year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "Rainfall": [208.43, 211.51, 182.04, 198.84, 227.74, 176.21, 203, 215.03, 188.92, 171.41]
}


# Combine the groundwater and rainfall data
combined_data = pd.merge(pd.DataFrame(groundwater_data), pd.DataFrame(rainfall_data), on='Year')



# Calculate correlation matrix
correlation_matrix = combined_data.corr()


# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()


# Create line plots for each groundwater parameter
parameters = combined_data.columns[1:-1]  # Exclude 'Year' and 'Rainfall'
for param in parameters:
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=combined_data['Rainfall'], y=combined_data[param])
    plt.xlabel("Rainfall")
    plt.ylabel(param)
    plt.title(f"Relationship between Rainfall and {param}")
    plt.show()


# Loop through groundwater parameters and calculate using equations
groundwater_params = ["Temperature", "ph", "Total dissolved solids", "Electrical conductivity", "Iron", "Nitrate", "Magnesium"]
   
        
# Loop through each parameter to calculate equation, coefficient, and intercept
for param in groundwater_params:
    
    # Split data into training (2012-2018) and testing (2019-2021) sets
    train_data = combined_data[combined_data['Year'] <= 2018]
    test_data = combined_data[combined_data['Year'] >= 2019]
    
    # Separate dependent variables (groundwater parameters) and independent variable (Rainfall) for training and testing
    y_train = train_data[param]
    X_train = train_data['Rainfall']
    
    y_test = test_data[param]
    X_test = test_data['Rainfall']
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train.values.reshape(-1, 1), y_train)
    
    # Predict groundwater parameters for the test data
    y_pred = model.predict(X_test.values.reshape(-1, 1)) 

    # Evaluate the model (MSE and R-squared)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    coefficient = model.coef_[0]
    print (f'{coefficient:.2f}')
    intercept = model.intercept_
    print (f'{intercept:.2f}')
   
    print()  # Print a newline 
    print(f"{param}:\n")
    print(f"Test Set MSE: {mse:.2f}, Test Set R-squared: {r2:.2f}")
    print(f"Coefficient (Slope): {coefficient:.3f}")
    print(f"Intercept: {intercept:.3f}\n")
    print(f"{param} = Coefficient * Rainfall + Intercept")
    
    param_rainfall_values = X_test.values
    print(f"Rainfall Test Years' data: {X_test.values[0]}mm, {X_test.values[1]}mm and {X_test.values[2]}mm")
    
    print(f'''
                
    For 1st {param} Data:
    {param} = {coefficient:.3f} * {X_test.values[0]} + {intercept:.3f} 

    For 2nd {param} Data:
    {param} = {coefficient:.3f} * {X_test.values[1]} + {intercept:.3f}
        
    For 3rd {param} Data:
    {param} = {coefficient:.3f} * {X_test.values[2]} + {intercept:.3f}

    ''')
    
    calculated_params = coefficient * param_rainfall_values  + intercept
    print(f"Calculated {param}:")
    
    for value in calculated_params:
        print(f"{value:.3f}")
   
    print()  # Print a newline after each parameter loop
    print('------------------------')  # Print a newline after each parameter loop
  
  
  
# # Github link:
# # https://github.com/Vix1209/Impact-of-rainfall-on-grndwater-parameters


