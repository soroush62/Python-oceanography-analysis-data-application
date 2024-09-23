# import pandas as pd
# import numpy as np
# import lightningchart as lc
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from scipy.interpolate import griddata
# import time

# # Load the license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load datasets
# beach_path = 'Dataset/beach.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# spot_path = 'Dataset/spot.csv'
# day_forecast_path = 'Dataset/day_forecast.csv'

# beach = pd.read_csv(beach_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# spot = pd.read_csv(spot_path)
# day_forecast = pd.read_csv(day_forecast_path)

# # Merge data to include latitude, longitude, and forecasting data
# merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')
# forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach', 
#                                                   'temperature', 'windspeed', 'winddirdegree', 
#                                                   'humidity', 'pressure']], on='idbeach')

# # Filter necessary columns for model
# X = final_data[['temperature', 'windspeed', 'winddirdegree', 'humidity', 'pressure']]
# y = final_data['sigheight']

# # Train/Test split for model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # RandomForest model for predicting sigheight
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Create the dashboard with one row and one column
# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=1, columns=1)

# # Create chart for the heatmap
# chart = dashboard.ChartXY(row_index=0, column_index=0, title='Predicted Wave Height Heatmap')
# heatmap = chart.add_heatmap_grid_series(
#     rows=50,  # Adjusted grid size
#     columns=50
# )

# # Set the heatmap boundaries and steps
# heatmap.set_start(x=merged_data['longitude'].min(), y=merged_data['latitude'].min())
# heatmap.set_end(x=merged_data['longitude'].max(), y=merged_data['latitude'].max())
# heatmap.set_step(
#     x=(merged_data['longitude'].max() - merged_data['longitude'].min()) / 50,
#     y=(merged_data['latitude'].max() - merged_data['latitude'].min()) / 50
# )

# # Customize axes titles
# chart.get_default_x_axis().set_title('Longitude')
# chart.get_default_y_axis().set_title('Latitude')

# # Generate random values and update the heatmap
# def generate_random_data():
#     random_data = {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max(), len(X)),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max(), len(X)),
#         'winddirdegree': np.random.uniform(X['winddirdegree'].min(), X['winddirdegree'].max(), len(X)),
#         'humidity': np.random.uniform(X['humidity'].min(), X['humidity'].max(), len(X)),
#         'pressure': np.random.uniform(X['pressure'].min(), X['pressure'].max(), len(X))
#     }
#     return pd.DataFrame(random_data)

# def update_heatmap():
#     for i in range(10):  # Simulate 10 updates to the heatmap
#         # Generate random feature values
#         random_features = generate_random_data()

#         # Predict sigheight for the generated features
#         predicted_sigheight = model.predict(random_features)

#         # Prepare data for heatmap plotting
#         x = final_data['longitude'].values
#         y = final_data['latitude'].values
#         z = predicted_sigheight

#         # Create grid for latitude and longitude
#         grid_lon = np.linspace(x.min(), x.max(), 50)
#         grid_lat = np.linspace(y.min(), y.max(), 50)
#         grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

#         # Interpolate sigheight for the grid points
#         grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')

#         # Fill NaN values with mean sigheight
#         grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)

#         # Debugging: Print the grid stats
#         print(f"Predicted SigHeight (first 10 values): {predicted_sigheight[:10]}")
#         print(f"Grid Wave Height Min: {np.nanmin(grid_wave_height)}")
#         print(f"Grid Wave Height Max: {np.nanmax(grid_wave_height)}")

#         # Set the interpolated sigheight values on the heatmap
#         heatmap.invalidate_intensity_values(grid_wave_height.tolist())
        
#         # Set a color palette based on the new predicted sigheight values
#         min_value = np.nanmin(grid_wave_height)
#         max_value = np.nanmax(grid_wave_height)

#         custom_palette = [
#             {"value": min_value, "color": lc.Color(0, 0, 255)},  # Blue for low height
#             {"value": np.percentile(grid_wave_height, 25), "color": lc.Color(0, 255, 255)},  # Cyan
#             {"value": np.median(grid_wave_height), "color": lc.Color(0, 255, 0)},  # Green
#             {"value": np.percentile(grid_wave_height, 75), "color": lc.Color(255, 255, 0)},  # Yellow
#             {"value": max_value, "color": lc.Color(255, 0, 0)}  # Red for high height
#         ]

#         # Apply the new color palette
#         heatmap.set_palette_colors(
#             steps=custom_palette,
#             look_up_property='value',
#             interpolate=True
#         )
        
#         # Sleep for a while to simulate dynamic updates
#         time.sleep(2)

# # Start the dashboard and heatmap update
# dashboard.open(live=True)
# update_heatmap()







#------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import lightningchart as lc
# import random
# import time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from scipy.interpolate import griddata

# # Load the license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load datasets
# beach_path = 'Dataset/beach.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# spot_path = 'Dataset/spot.csv'
# day_forecast_path = 'Dataset/day_forecast.csv'

# beach = pd.read_csv(beach_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# spot = pd.read_csv(spot_path)
# day_forecast = pd.read_csv(day_forecast_path)

# # Merge data to include latitude, longitude, and forecasting data
# merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')
# forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach', 
#                                                   'temperature', 'windspeed', 'winddirdegree', 
#                                                   'humidity', 'pressure']], on='idbeach')

# # Filter necessary columns for model
# X = final_data[['temperature', 'windspeed', 'winddirdegree', 'humidity', 'pressure']]
# y = final_data['sigheight']

# # Train/Test split for model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # RandomForest model for predicting sigheight
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Function to generate random weather data for prediction and humidity chart
# def generate_random_weather_data(num_entries):
#     random_data = []
#     for i in range(num_entries):
#         random_data.append({
#             'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
#             'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
#             'winddirdegree': np.random.uniform(X['winddirdegree'].min(), X['winddirdegree'].max()),
#             'humidity': np.random.uniform(X['humidity'].min(), X['humidity'].max()),
#             'pressure': np.random.uniform(X['pressure'].min(), X['pressure'].max())
#         })
#     return pd.DataFrame(random_data)

# # Prepare data for heatmap plotting
# def update_heatmap():
#     random_weather_df = generate_random_weather_data(len(final_data))
#     predicted_sigheight = model.predict(random_weather_df)
    
#     # Update final data with predicted sigheight
#     final_data['predicted_sigheight'] = predicted_sigheight
    
#     # Create grid for latitude and longitude
#     x = final_data['longitude'].values
#     y = final_data['latitude'].values
#     z = final_data['predicted_sigheight'].values
#     grid_lon = np.linspace(x.min(), x.max(), 50)
#     grid_lat = np.linspace(y.min(), y.max(), 50)
#     grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
#     grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')
#     grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)
    
#     # Update heatmap
#     heatmap.invalidate_intensity_values(grid_wave_height.tolist())

# # Create the dashboard with two rows
# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=1)

# # 1st Row: Humidity Time-Series Chart
# chart_humidity = dashboard.ChartXY(row_index=0, column_index=0, title="Humidity Changes Over Time")
# legend = chart_humidity.add_legend()

# # Generate humidity data over time for each beach
# time_points = list(range(1, 51))  # 50 time points
# unique_beaches = final_data['idbeach'].unique()

# humidity_series = []

# for i, beach_id in enumerate(unique_beaches[:5]):  # Show 5 beaches max
#     humidity_values = [np.random.uniform(50, 100) for _ in time_points]
#     axis_y = chart_humidity.add_y_axis(stack_index=i)
#     axis_y.set_margins(15 if i > 0 else 0, 15 if i < 3 else 0)
#     axis_y.set_title_rotation(45)
#     axis_y.set_title(f'Beach {beach_id}')
    
#     series = chart_humidity.add_line_series(y_axis=axis_y, data_pattern='ProgressiveX')
#     series.add(time_points, humidity_values)
#     series.set_name(f'Beach {beach_id}')
#     legend.add(series)
    
#     humidity_series.append(series)

# chart_humidity.get_default_x_axis().set_title('Time Points')
# chart_humidity.get_default_y_axis().set_title('Humidity (%)')

# # 2nd Row: Dynamic Heatmap for Predicted Wave Heights
# chart_heatmap = dashboard.ChartXY(row_index=1, column_index=0, title='Predicted Wave Height Heatmap')

# heatmap = chart_heatmap.add_heatmap_grid_series(
#     rows=50, 
#     columns=50
# )
# heatmap.set_start(x=final_data['longitude'].min(), y=final_data['latitude'].min())
# heatmap.set_end(x=final_data['longitude'].max(), y=final_data['latitude'].max())
# heatmap.set_step(x=(final_data['longitude'].max() - final_data['longitude'].min()) / 50,
#                  y=(final_data['latitude'].max() - final_data['latitude'].min()) / 50)

# # Set a color palette for the heatmap
# custom_palette = [
#     {"value": 0, "color": lc.Color(0, 0, 255)},   # Blue for low wave height
#     {"value": 0.5, "color": lc.Color(0, 255, 0)},  # Green for mid wave height
#     {"value": 1, "color": lc.Color(255, 0, 0)}    # Red for high wave height
# ]
# heatmap.set_palette_colors(
#     steps=custom_palette,
#     look_up_property='value',
#     interpolate=True
# )

# # Initial update of heatmap
# update_heatmap()

# # Function to update the dashboard dynamically with new data
# def update_dashboard():
#     while True:
#         # Generate new random humidity data and update humidity chart
#         for series in humidity_series:
#             new_humidity_values = [np.random.uniform(50, 100) for _ in time_points]
#             series.clear().add(time_points, new_humidity_values)
        
#         # Update heatmap with new predicted wave heights
#         update_heatmap()

#         time.sleep(2)  # Update every 2 seconds

# # Open the dashboard and start updating
# dashboard.open(live=True)
# update_dashboard()








#------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import lightningchart as lc
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from scipy.interpolate import griddata
# import random
# import time

# # Load the license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load datasets
# beach_path = 'Dataset/beach.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# spot_path = 'Dataset/spot.csv'
# day_forecast_path = 'Dataset/day_forecast.csv'

# beach = pd.read_csv(beach_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# spot = pd.read_csv(spot_path)
# day_forecast = pd.read_csv(day_forecast_path)

# # Merge data to include latitude, longitude, and forecasting data
# merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')
# forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach', 
#                                                   'temperature', 'windspeed', 'winddirdegree', 
#                                                   'humidity', 'pressure']], on='idbeach')

# # Filter necessary columns for model
# X = final_data[['temperature', 'windspeed', 'winddirdegree', 'humidity', 'pressure']]
# y = final_data['sigheight']

# # Train/Test split for model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # RandomForest model for predicting sigheight
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Create the dashboard with two rows (for heatmap and humidity chart)
# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=1)

# # --------------------------- Row 1: Stacked Humidity Chart (like the example) --------------------------- #
# humidity_chart = dashboard.ChartXY(row_index=0, column_index=0, title="Real-Time Humidity Changes Over Time")
# humidity_chart.get_default_y_axis().dispose()  # Dispose the default Y-axis

# legend = humidity_chart.add_legend()

# # Generate time axis for the x-axis (simulating in milliseconds for progressive charting)
# time_step = 0
# x_axis = humidity_chart.get_default_x_axis()
# x_axis.set_title('Time (ms)')

# # Prepare series for each beach with stacked Y-axes
# unique_beaches = beach['idbeach'].unique()
# humidity_series = {}

# for i, beach_id in enumerate(unique_beaches[:5]):  # Limit to the first 5 beaches for clarity
#     axis_y = humidity_chart.add_y_axis(stack_index=i)
#     axis_y.set_margins(15 if i > 0 else 0, 15 if i < len(unique_beaches[:5])-1 else 0)  
#     axis_y.set_title(f'Beach {int(beach_id)}')
#     axis_y.set_title_rotation(45)
#     series = humidity_chart.add_line_series(y_axis=axis_y, data_pattern='ProgressiveX')
#     series.set_name(f'Beach {int(beach_id)}')
#     humidity_series[beach_id] = series
#     legend.add(series)

# # --------------------------- Row 2: Dynamic Heatmap for Sigheight --------------------------- #
# chart = dashboard.ChartXY(row_index=1, column_index=0, title='Predicted Wave Height Heatmap')
# heatmap = chart.add_heatmap_grid_series(rows=50, columns=50)

# # Set the heatmap boundaries and steps
# heatmap.set_start(x=merged_data['longitude'].min(), y=merged_data['latitude'].min())
# heatmap.set_end(x=merged_data['longitude'].max(), y=merged_data['latitude'].max())
# heatmap.set_step(
#     x=(merged_data['longitude'].max() - merged_data['longitude'].min()) / 50,
#     y=(merged_data['latitude'].max() - merged_data['latitude'].min()) / 50
# )
# # Customize axes titles
# chart.get_default_x_axis().set_title('Longitude')
# chart.get_default_y_axis().set_title('Latitude')

# # Generate random humidity values and update the heatmap
# def generate_random_data():
#     random_data = {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max(), len(X)),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max(), len(X)),
#         'winddirdegree': np.random.uniform(X['winddirdegree'].min(), X['winddirdegree'].max(), len(X)),
#         'humidity': np.random.uniform(X['humidity'].min(), X['humidity'].max(), len(X)),
#         'pressure': np.random.uniform(X['pressure'].min(), X['pressure'].max(), len(X))
#     }
#     return pd.DataFrame(random_data)

# def update_dashboard():
#     global time_step
    
#     while True:
#         # Generate random feature values for each beach and update humidity chart
#         random_features = generate_random_data()
#         new_sigheight = model.predict(random_features)

#         # Update the humidity chart for each beach
#         for beach_id in unique_beaches[:5]:  # Only updating the first 5 beaches
#             new_humidity = random.uniform(30, 70)  # Generate random humidity between 30-70
#             humidity_series[beach_id].add(time_step, new_humidity)

#         # Update the heatmap with the new sigheight predictions
#         x = final_data['longitude'].values
#         y = final_data['latitude'].values
#         z = new_sigheight

#         grid_lon = np.linspace(x.min(), x.max(), 50)
#         grid_lat = np.linspace(y.min(), y.max(), 50)
#         grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

#         # Interpolate sigheight for the grid points
#         grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')

#         # Fill NaN values with mean sigheight
#         grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)

#         # Set the interpolated sigheight values on the heatmap
#         heatmap.invalidate_intensity_values(grid_wave_height.tolist())
        
#         # Set a color palette based on the new predicted sigheight values
#         min_value = np.nanmin(grid_wave_height)
#         max_value = np.nanmax(grid_wave_height)

#         custom_palette = [
#             {"value": 0.8, "color": lc.Color(0, 0, 255)},  # Blue for low height
#             {"value": 0.9, "color": lc.Color(0, 255, 0)},  # Green
#             {"value": 1, "color": lc.Color(255, 255, 0)},  # Yellow
#             {"value": 2, "color": lc.Color(255, 0, 0)}  # Red for high height
#         ]


#         # Apply the new color palette
#         heatmap.set_palette_colors(
#             steps=custom_palette,
#             look_up_property='value',
#             interpolate=True
#         )

#         # Increment time step and wait for 1 second to simulate dynamic updates
#         time_step += 100
#         time.sleep(1)

# # Open the dashboard and start real-time updates
# dashboard.open(live=True)
# update_dashboard()











# import pandas as pd
# import numpy as np
# import lightningchart as lc
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from scipy.interpolate import griddata
# import random
# import time

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# beach_path = 'Dataset/beach.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# spot_path = 'Dataset/spot.csv'
# day_forecast_path = 'Dataset/day_forecast.csv'

# beach = pd.read_csv(beach_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# spot = pd.read_csv(spot_path)
# day_forecast = pd.read_csv(day_forecast_path)

# merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')
# forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach', 
#                                                   'temperature', 'windspeed', 'winddirdegree', 
#                                                   'humidity', 'pressure']], on='idbeach')
# beach_num=final_data['idbeach'].nunique()
# X = final_data[['temperature', 'windspeed', 'winddirdegree', 'humidity', 'pressure']]
# y = final_data['sigheight']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=1)

# # --------------------------- Row 1: Chart showing average of features over time --------------------------- #
# feature_chart = dashboard.ChartXY(row_index=0, column_index=0, title="Average Feature Values Over Time")
# feature_chart.get_default_y_axis().dispose()  

# legend = feature_chart.add_legend()

# time_step = 0
# x_axis = feature_chart.get_default_x_axis()
# x_axis.set_title('Time (ms)')

# series_dict = {}

# features = ['temperature', 'windspeed', 'humidity', 'pressure']
# for i, feature in enumerate(features):
#     axis_y = feature_chart.add_y_axis(stack_index=i)
#     axis_y.set_margins(15 if i > 0 else 0, 15 if i < len(features) - 1 else 0)  
#     axis_y.set_title(f'{feature.capitalize()}')
#     axis_y.set_title_rotation(45)
#     series = feature_chart.add_line_series(y_axis=axis_y, data_pattern='ProgressiveX')
#     series.set_name(f'Average {feature.capitalize()}')
#     series_dict[feature] = series
#     legend.add(series)

# # --------------------------- Row 2: Dynamic Heatmap for Sigheight --------------------------- #
# chart = dashboard.ChartXY(row_index=1, column_index=0, title='Predicted Wave Height Heatmap')
# heatmap = chart.add_heatmap_grid_series(rows=50, columns=50)

# heatmap.set_start(x=final_data['longitude'].min(), y=final_data['latitude'].min())
# heatmap.set_end(x=final_data['longitude'].max(), y=final_data['latitude'].max())
# # heatmap.set_step(
# #     x=(merged_data['longitude'].max() - merged_data['longitude'].min()) / 50,
# #     y=(merged_data['latitude'].max() - merged_data['latitude'].min()) / 50
# # )

# chart.get_default_x_axis().set_title('Longitude')
# chart.get_default_y_axis().set_title('Latitude')

# def generate_random_data():
#     random_data = {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max(), beach_num),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max(), beach_num),
#         'winddirdegree': np.random.uniform(X['winddirdegree'].min(), X['winddirdegree'].max(), beach_num),
#         'humidity': np.random.uniform(X['humidity'].min(), X['humidity'].max(), beach_num),
#         'pressure': np.random.uniform(X['pressure'].min(), X['pressure'].max(), beach_num)
#     }
#     return pd.DataFrame(random_data)

# def update_dashboard():
#     global time_step
    
#     while True:
#         random_features = generate_random_data()
#         new_sigheight = model.predict(random_features)

#         for feature in features:
#             avg_value = random_features[feature].mean()  
#             series_dict[feature].add(time_step, avg_value)

#         x = final_data['longitude'].values
#         y = final_data['latitude'].values
#         z = new_sigheight

#         grid_lon = np.linspace(x.min(), x.max(), 50)
#         grid_lat = np.linspace(y.min(), y.max(), 50)
#         grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

#         grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')

#         grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)

#         heatmap.invalidate_intensity_values(grid_wave_height.tolist())
        
#         min_value = np.nanmin(grid_wave_height)
#         max_value = np.nanmax(grid_wave_height)

#         custom_palette = [
#             {"value": 0.8, "color": lc.Color(0, 0, 255)},  # Blue
#             {"value": 0.9, "color": lc.Color(0, 255, 0)},  # Green
#             {"value": 1, "color": lc.Color(255, 255, 0)},  # Yellow
#             {"value": 2, "color": lc.Color(255, 0, 0)}  # Red 
#         ]

#         heatmap.set_palette_colors(
#             steps=custom_palette,
#             look_up_property='value',
#             interpolate=True
#         )

#         time_step += 100
#         time.sleep(1)

# dashboard.open(live=True)
# update_dashboard()






import pandas as pd
import numpy as np
import lightningchart as lc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import random
import time

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

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
        
        # min_value = np.nanmin(grid_wave_height)
        # max_value = np.nanmax(grid_wave_height)

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


