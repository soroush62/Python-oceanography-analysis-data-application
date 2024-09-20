import lightningchart as lc
import pandas as pd
import numpy as np

# Set up the license
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the datasets
beach_path = 'Dataset/beach.csv'
hour_forecast_path = 'Dataset/hour_forecast.csv'
spot_path = 'Dataset/spot.csv'
day_forecast_path = 'Dataset/day_forecast.csv'

beach = pd.read_csv(beach_path)
hour_forecast = pd.read_csv(hour_forecast_path)
spot = pd.read_csv(spot_path)
day_forecast = pd.read_csv(day_forecast_path)

# Merge datasets to obtain temperature, windspeed, and time values
merged_data = pd.merge(spot, beach[['idbeach', 'name']], on='idbeach')
forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'time', 'temperature', 'windspeed', 'idbeach']], on='idbeach')

# Group by time to calculate min and max values for temperature and windspeed
temp_wind_data = final_data.groupby('time').agg({
    'temperature': ['min', 'max'],
    'windspeed': ['min', 'max']
}).reset_index()

# Extract min and max values for temperature and windspeed
time_values = np.arange(len(temp_wind_data))  # Simulate time values with integers for simplicity
temp_min = temp_wind_data['temperature']['min'].tolist()
temp_max = temp_wind_data['temperature']['max'].tolist()
wind_min = temp_wind_data['windspeed']['min'].tolist()
wind_max = temp_wind_data['windspeed']['max'].tolist()

# Create an Area Range Chart in LightningChart
chart = lc.ChartXY(title="Temperature vs. Windspeed Correlation (Range over Time)", theme=lc.Themes.Dark)

# Create an Area Range Series for Temperature
temp_series = chart.add_area_range_series()
temp_series.set_name('Temperature Range (Min to Max)')
temp_series.add_arrays_high_low(high=temp_max, low=temp_min, start=0, step=1)
temp_series.set_high_fill_color(lc.Color(0, 191, 255, 150))  # Light blue for temperature
temp_series.set_low_fill_color(lc.Color(0, 191, 255, 150))

# Create an Area Range Series for Windspeed
wind_series = chart.add_area_range_series()
wind_series.set_name('Windspeed Range (Min to Max)')
wind_series.add_arrays_high_low(high=wind_max, low=wind_min, start=0, step=1)
wind_series.set_high_fill_color(lc.Color(50, 205, 50, 150))  # Light green for windspeed
wind_series.set_low_fill_color(lc.Color(50, 205, 50, 150))

# Customize the axes titles
chart.get_default_x_axis().set_title('Time')
chart.get_default_y_axis().set_title('Value')

# Open the chart directly without invoking serialization
chart.open()
