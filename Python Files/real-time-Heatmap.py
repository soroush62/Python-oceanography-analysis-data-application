import pandas as pd
import numpy as np
import lightningchart as lc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import random
import time

lc.set_license(open('../license-key').read())

beach_path = 'Dataset/beach.csv'
hour_forecast_path = 'Dataset/hour_forecast.csv'
spot_path = 'Dataset/spot.csv'
day_forecast_path = 'Dataset/day_forecast.csv'

beach = pd.read_csv(beach_path)
hour_forecast = pd.read_csv(hour_forecast_path)
spot = pd.read_csv(spot_path)
day_forecast = pd.read_csv(day_forecast_path)

merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')
forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach', 
                                                  'temperature', 'windspeed', 'winddirdegree', 
                                                  'humidity', 'pressure']], on='idbeach')

beach_num = final_data['idbeach'].nunique()
beach_subset = beach.iloc[:7]  
X = final_data[['temperature', 'windspeed', 'winddirdegree', 'humidity', 'pressure']]
y = final_data['sigheight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=1)

# --------------------------- Row 1: Chart showing average of features over time --------------------------- #
feature_chart = dashboard.ChartXY(row_index=0, column_index=0, title="Average Feature Values Over Time")
feature_chart.get_default_y_axis().dispose()  

legend = feature_chart.add_legend()

time_step = 0
x_axis = feature_chart.get_default_x_axis()
x_axis.set_title('Time (ms)')

series_dict = {}

features = ['temperature', 'windspeed', 'humidity', 'pressure']
for i, feature in enumerate(features):
    axis_y = feature_chart.add_y_axis(stack_index=i)
    axis_y.set_margins(15 if i > 0 else 0, 15 if i < len(features) - 1 else 0)  
    axis_y.set_title(f'{feature.capitalize()}')
    axis_y.set_title_rotation(45)
    series = feature_chart.add_line_series(y_axis=axis_y, data_pattern='ProgressiveX')
    series.set_name(f'Average {feature.capitalize()}')
    series_dict[feature] = series
    legend.add(series)

# --------------------------- Row 2: Dynamic Heatmap for Sigheight --------------------------- #
chart = dashboard.ChartXY(row_index=1, column_index=0, title='Predicted Wave Height Heatmap')
heatmap = chart.add_heatmap_grid_series(rows=50, columns=50)

heatmap.set_start(x=beach_subset['longitude'].min(), y=beach_subset['latitude'].min())
heatmap.set_end(x=beach_subset['longitude'].max(), y=beach_subset['latitude'].max())
heatmap.set_step(
    x=(beach_subset['longitude'].max() - beach_subset['longitude'].min()) / 50,
    y=(beach_subset['latitude'].max() - beach_subset['latitude'].min()) / 50
)

chart.get_default_x_axis().set_title('Longitude')
chart.get_default_y_axis().set_title('Latitude')

def generate_random_data():
    random_data = {
        'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max(), beach_num),
        'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max(), beach_num),
        'winddirdegree': np.random.uniform(X['winddirdegree'].min(), X['winddirdegree'].max(), beach_num),
        'humidity': np.random.uniform(X['humidity'].min(), X['humidity'].max(), beach_num),
        'pressure': np.random.uniform(X['pressure'].min(), X['pressure'].max(), beach_num)
    }
    return pd.DataFrame(random_data)

def update_dashboard():
    global time_step
    
    while True:
        random_features = generate_random_data()
        new_sigheight = model.predict(random_features)

        for feature in features:
            avg_value = random_features[feature].mean()  
            series_dict[feature].add(time_step, avg_value)

        x = beach_subset['longitude'].values  
        y = beach_subset['latitude'].values   
        z = new_sigheight                     

        grid_lon = np.linspace(x.min(), x.max(), 50)
        grid_lat = np.linspace(y.min(), y.max(), 50)
        grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

        grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')

        grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)

        heatmap.invalidate_intensity_values(grid_wave_height.tolist())
        
        custom_palette = [
            {"value": 1.0, "color": lc.Color(0, 0, 255)},  # Blue 
            {"value": 1.1, "color": lc.Color(255, 255, 0)},  # Yellow
            {"value": 1.3, "color": lc.Color(255, 0, 0)}  # Red
        ]

        heatmap.set_palette_colors(
            steps=custom_palette,
            look_up_property='value',
            interpolate=True
        )

        time_step += 100
        time.sleep(1)

dashboard.open(live=True)
update_dashboard()