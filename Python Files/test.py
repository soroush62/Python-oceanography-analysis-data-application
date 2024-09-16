import pandas as pd
import numpy as np
import lightningchart as lc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set LightningChart license
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load your dataset (modify the path accordingly)
data = pd.read_csv('Dataset/hour_forecast.csv')

# Features and target
X = data[['temperature', 'windspeed']]
y = data['sigheight']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Function to predict sigheight
def predict_sigheight(temp, windspeed):
    scaled_features = scaler.transform([[temp, windspeed]])
    return model.predict(scaled_features)[0]

# Create Dashboard with 2 rows and 2 columns
dashboard = lc.Dashboard(rows=2, columns=2, theme=lc.Themes.Dark)

# Instead of sliders, we'll use a form of user input. For now, let's assume default values for Temperature and Windspeed.
default_temperature = 20  # Hard-coded
default_windspeed = 30  # Hard-coded

# Add Temperature TextBox
temp_textbox = dashboard.ChartXY(row_index=0, column_index=0)
temp_textbox.set_title(f'Temperature (°C): {default_temperature}')

# Add Windspeed TextBox
wind_textbox = dashboard.ChartXY(row_index=0, column_index=1)
wind_textbox.set_title(f'Windspeed (m/s): {default_windspeed}')

# Add Sigheight Gauge
sigheight_gauge = dashboard.GaugeChart(row_index=1, column_index=0, column_span=2)
sigheight_gauge.set_title('Predicted Sigheight')
sigheight_gauge.set_interval(0, 2)  # Adjust according to expected sigheight range
sigheight_gauge.set_value(0)  # Default starting value

# Function to update prediction dynamically
def update_prediction(temp, windspeed):
    predicted_sigheight = predict_sigheight(temp, windspeed)
    # Update the Sigheight Gauge
    sigheight_gauge.set_value(predicted_sigheight)

# For now, update prediction with default values
update_prediction(default_temperature, default_windspeed)

# Open the dashboard
dashboard.open()
