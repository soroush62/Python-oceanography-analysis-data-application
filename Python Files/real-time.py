# import pandas as pd
# import numpy as np
# import lightningchart as lc
# import random
# import time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Set up your LightningChart license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load your dataset
# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# # Random data generation (for simulation purposes)
# X = data[['temperature', 'windspeed']]
# y = data['sigheight']

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model for predicting `sigheight`
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Create the dashboard
# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

# # Left gauge: Random temperature
# temperature_gauge = dashboard.GaugeChart(row_index=0, column_index=0)
# temperature_gauge.set_title('Temperature')
# temperature_gauge.set_angle_interval(start=225, end=-45)
# temperature_gauge.set_interval(start=-10, end=50)
# temperature_gauge.set_value_indicators([
#     {'start': -10, 'end': 0, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 0, 'end': 10, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 20, 'end': 30, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 30, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# temperature_gauge.set_bar_thickness(30)
# temperature_gauge.set_value_indicator_thickness(8)

# # Middle chart: 3D surface simulating wind direction
# wind_direction_series = dashboard.SpiderChart(row_index=0, column_index=1)
# wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
# series = wind_direction_series.add_series().add_points([
#     {'axis': 'N', 'value': 0},
#     {'axis': 'NE', 'value': 0},
#     {'axis': 'E', 'value': 0},
#     {'axis': 'SE', 'value': 0},
#     {'axis': 'S', 'value': 0},
#     {'axis': 'SW', 'value': 0},
#     {'axis': 'W', 'value': 0},
#     {'axis': 'NW', 'value': 0},
# ])

# # Right gauge: Random windspeed
# windspeed_gauge = dashboard.GaugeChart(row_index=0, column_index=2)
# windspeed_gauge.set_title('Windspeed')
# windspeed_gauge.set_angle_interval(start=225, end=-45)
# windspeed_gauge.set_interval(start=0, end=100)
# windspeed_gauge.set_value_indicators([
#     {'start': 0, 'end': 20, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 20, 'end': 40, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 40, 'end': 60, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 60, 'end': 80, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 80, 'end': 100, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# windspeed_gauge.set_bar_thickness(30)
# windspeed_gauge.set_value_indicator_thickness(8)

# # Bottom surface chart: Predict `sigheight` with 3D surface chart
# sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
# sigheight_chart.set_title('Predicted Sigheight Over Time')
# surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=1, scroll_dimension='columns')  # Adjusted to 1 row
# surface.set_min_max_palette_colors(
#     min_value=y.min(),
#     max_value=y.max(),
#     min_color=lc.Color(0, 0, 255),  # Blue
#     max_color=lc.Color(255, 0, 0)   # Red
# )

# # Generate random temperature, windspeed, and wind direction data
# def generate_random_weather_data():
#     return {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
#         'wind_direction': [random.uniform(0, 100) for _ in wind_directions]  # Random wind speed in each direction
#     }

# import pandas as pd
# import numpy as np
# import lightningchart as lc
# import random
# import time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Set up your LightningChart license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load your dataset
# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# # Random data generation (for simulation purposes)
# X = data[['temperature', 'windspeed']]
# y = data['sigheight']

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model for predicting `sigheight`
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# # Create the dashboard
# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

# # Left gauge: Random temperature
# temperature_gauge = dashboard.GaugeChart(row_index=0, column_index=0)
# temperature_gauge.set_title('Temperature')
# temperature_gauge.set_angle_interval(start=225, end=-45)
# temperature_gauge.set_interval(start=-10, end=50)
# temperature_gauge.set_value_indicators([
#     {'start': -10, 'end': 0, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 0, 'end': 10, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 20, 'end': 30, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 30, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# temperature_gauge.set_bar_thickness(30)
# temperature_gauge.set_value_indicator_thickness(8)

# # Middle chart: Spider chart for wind direction
# spider_chart = dashboard.SpiderChart(row_index=0, column_index=1)
# spider_chart.set_title('Spider Chart')
# wind_direction_series = spider_chart.add_series()

# # Right gauge: Random windspeed
# windspeed_gauge = dashboard.GaugeChart(row_index=0, column_index=2)
# windspeed_gauge.set_title('Windspeed')
# windspeed_gauge.set_angle_interval(start=225, end=-45)
# windspeed_gauge.set_interval(start=0, end=100)
# windspeed_gauge.set_value_indicators([
#     {'start': 0, 'end': 20, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 20, 'end': 40, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 40, 'end': 60, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 60, 'end': 80, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 80, 'end': 100, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# windspeed_gauge.set_bar_thickness(30)
# windspeed_gauge.set_value_indicator_thickness(8)

# # Bottom surface chart: Predict `sigheight` with 3D surface chart
# sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
# sigheight_chart.set_title('Predicted Sigheight Over Time')
# surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=100, scroll_dimension='columns')
# surface.set_min_max_palette_colors(
#     min_value=-50,
#     max_value=50,
#     min_color=lc.Color('#00ffff'),
#     max_color=lc.Color('#ffff00'),
#     look_up_property='y'
# )
# surface.set_wireframe_stroke(1, color=lc.Color(0, 0, 0, 255))

# # Generate random temperature, windspeed, and wind direction data
# def generate_random_weather_data():
#     return {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
#         'wind_direction': [random.uniform(0, 100) for _ in wind_directions]  # Random wind speed in each direction
#     }

# # Function to update dashboard dynamically
# def update_dashboard():
#     time_values = []
#     start_time = time.time()
#     i = 0
#     x = 0
    
#     while i < 1000:  
#         random_weather = generate_random_weather_data()
#         random_weather_df = pd.DataFrame([random_weather])

#         predicted_sigheight = model.predict(random_weather_df[['temperature', 'windspeed']])[0]

#         current_time = time.time() - start_time
#         time_values.append(current_time)
#         i += 1
#         x = x + (random.random() * 2) - 1

#         # Update the temperature gauge and windspeed gauge
#         temperature_gauge.set_value(random_weather['temperature'])
#         windspeed_gauge.set_value(random_weather['windspeed'])

#         # Overwrite the wind direction points with new random values
#         wind_direction_series.add_points([  # Add new random points
#             {'axis': direction, 'value': speed}
#             for direction, speed in zip(wind_directions, random_weather['wind_direction'])
#         ])

#         # Debugging: Print the predicted sigheight value to make sure it is within expected range
#         print(f"Predicted Sigheight: {predicted_sigheight}")
#         print(f"Temperature: {random_weather['temperature']})")
#         print(f"Windspeed: {random_weather['windspeed']})")
#         print(f"Wind Direction: {random_weather['wind_direction']})")

#         # Update the predicted sigheight surface chart
#         # Ensure predicted_sigheight is added as a 2D array (list of lists)
#         grid = np.full((1, 100), predicted_sigheight)  # Creating a 2D array filled with predicted_sigheight values
#         surface.add_values(grid.tolist())  # Adding grid values to the surface chart

#         time.sleep(4)

# dashboard.open(live=True)
# update_dashboard()







# import pandas as pd
# import numpy as np
# import lightningchart as lc
# import random
# import time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# X = data[['temperature', 'windspeed',]]
# y = data['sigheight']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

# temperature_gauge = dashboard.GaugeChart(row_index=0, column_index=0)
# temperature_gauge.set_title('Temperature')
# temperature_gauge.set_angle_interval(start=225, end=-45)
# temperature_gauge.set_interval(start=-10, end=50)
# temperature_gauge.set_value_indicators([
#     {'start': -10, 'end': 0, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 0, 'end': 10, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 20, 'end': 30, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 30, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# temperature_gauge.set_bar_thickness(30)
# temperature_gauge.set_value_indicator_thickness(8)

# spider_chart = dashboard.SpiderChart(row_index=0, column_index=1)
# spider_chart.set_title('Spider Chart')
# wind_direction_series = spider_chart.add_series()

# windspeed_gauge = dashboard.GaugeChart(row_index=0, column_index=2)
# windspeed_gauge.set_title('Windspeed')
# windspeed_gauge.set_angle_interval(start=225, end=-45)
# windspeed_gauge.set_interval(start=0, end=100)
# windspeed_gauge.set_value_indicators([
#     {'start': 0, 'end': 20, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 20, 'end': 40, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 40, 'end': 60, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 60, 'end': 80, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 80, 'end': 100, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# windspeed_gauge.set_bar_thickness(30)
# windspeed_gauge.set_value_indicator_thickness(8)

# sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
# sigheight_chart.set_title('Predicted Sigheight Over Time')
# axis = sigheight_chart.get_default_x_axis()
# axis.set_scroll_strategy('fitting')
# axis.set_tick_strategy('Numeric')
# surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=100, scroll_dimension='columns')
# surface.set_min_max_palette_colors(
#     min_value=-50,
#     max_value=50,
#     min_color=lc.Color('#00ffff'),
#     max_color=lc.Color('#ffff00'),
#     look_up_property='y'
# )
# surface.set_wireframe_stroke(1, color=lc.Color(0, 128, 255))

# def generate_random_weather_data():
#     wind_direction_values = [0] * len(wind_directions)
#     selected_direction = random.randint(0, len(wind_directions) - 1)
#     wind_direction_values[selected_direction] = random.uniform(0, 100)  

#     return {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
#         'wind_direction': wind_direction_values 
#     }

# def update_dashboard():
#     time_values = []
#     start_time = time.time()
#     i = 0
#     x = 0
    
#     while i < 1000:  
#         random_weather = generate_random_weather_data()
#         random_weather_df = pd.DataFrame([random_weather])

#         predicted_sigheight = model.predict(random_weather_df[['temperature', 'windspeed']])[0]

#         current_time = time.time() - start_time
#         time_values.append(current_time)
#         i += 1
#         x = x + (random.random() * 2) - 1

#         temperature_gauge.set_value(random_weather['temperature'])
#         windspeed_gauge.set_value(random_weather['windspeed'])

#         wind_direction_series.add_points([
#             {'axis': direction, 'value': value}
#             for direction, value in zip(wind_directions, random_weather['wind_direction'])
#         ])

#         print(f"Predicted Sigheight: {predicted_sigheight}")

#         grid = np.full((1, 100), predicted_sigheight) 
#         surface.add_values(grid.tolist())  

#         time.sleep(1)

# dashboard.open(live=True)
# update_dashboard()
















# import pandas as pd
# import numpy as np
# import lightningchart as lc
# import random
# import time
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# X = data[['temperature', 'windspeed']]
# y = data['sigheight']

# wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
# encoder = OneHotEncoder(sparse=False)
# random_wind_directions = np.random.choice(wind_directions, size=len(data))
# encoded_wind_directions = pd.DataFrame(
#     encoder.fit_transform(random_wind_directions.reshape(-1, 1)),
#     columns=encoder.get_feature_names_out(['wind_direction'])
# )

# X = pd.concat([X, encoded_wind_directions], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)

# dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

# temperature_gauge = dashboard.GaugeChart(row_index=0, column_index=0)
# temperature_gauge.set_title('Temperature')
# temperature_gauge.set_angle_interval(start=225, end=-45)
# temperature_gauge.set_interval(start=0, end=50)
# temperature_gauge.set_unit_label('Celsius')
# temperature_gauge.set_unit_label_font(size=25, weight='bold')
# temperature_gauge.set_value_indicators([
#     {'start': 0, 'end': 10, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 20, 'end': 30, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 30, 'end': 40, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 40, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# temperature_gauge.set_bar_thickness(30)
# temperature_gauge.set_value_indicator_thickness(8)

# spider_chart = dashboard.SpiderChart(row_index=0, column_index=1)
# spider_chart.set_title('Wind Direction')
# wind_direction_series = spider_chart.add_series()

# windspeed_gauge = dashboard.GaugeChart(row_index=0, column_index=2)
# windspeed_gauge.set_title('Windspeed')
# windspeed_gauge.set_angle_interval(start=225, end=-45)
# windspeed_gauge.set_interval(start=0, end=50)
# windspeed_gauge.set_unit_label('m/s')
# windspeed_gauge.set_unit_label_font(size=25, weight='bold')
# windspeed_gauge.set_value_indicators([
#     {'start': 0, 'end': 10, 'color': lc.Color(0, 0, 255)},  # Blue
#     {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 255)},  # Cyan
#     {'start': 20, 'end': 30, 'color': lc.Color(0, 255, 0)},  # Green
#     {'start': 30, 'end': 40, 'color': lc.Color(255, 255, 0)},  # Yellow
#     {'start': 40, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
# ])
# windspeed_gauge.set_bar_thickness(30)
# windspeed_gauge.set_value_indicator_thickness(8)

# sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
# sigheight_chart.set_title('Predicted Sigheight Over Time')
# sigheight_chart.set_bounding_box(x=9.0, y=1.0, z=1)
# axis = sigheight_chart.get_default_x_axis()
# axis.set_scroll_strategy('fitting')
# axis.set_tick_strategy('Numeric')
# surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=100, scroll_dimension='columns')
# surface.set_min_max_palette_colors(
#     min_value=-50,
#     max_value=50,
#     min_color=lc.Color('#00ffff'),
#     max_color=lc.Color('#ffff00'),
#     look_up_property='y'
# )
# surface.set_wireframe_stroke(1, color=lc.Color(0, 128, 255))

# def generate_random_weather_data():
#     wind_direction_values = [0] * len(wind_directions)
#     selected_direction = random.randint(0, len(wind_directions) - 1)
#     wind_direction_values[selected_direction] = random.uniform(0, 100) 

#     return {
#         'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
#         'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
#         'wind_direction': wind_direction_values 
#     }

# def update_dashboard():
#     time_values = []
#     start_time = time.time()
#     i = 0
#     x = 0
    
#     while i < 1000:  
#         random_weather = generate_random_weather_data()
#         random_weather_df = pd.DataFrame([random_weather])

#         wind_direction_encoded = pd.DataFrame([random_weather['wind_direction']], columns=encoder.get_feature_names_out(['wind_direction']))
#         random_weather_df = pd.concat([random_weather_df[['temperature', 'windspeed']], wind_direction_encoded], axis=1)

#         predicted_sigheight = model.predict(random_weather_df)[0]

#         current_time = time.time() - start_time
#         time_values.append(current_time)
#         i += 1
#         x = x + (random.random() * 2) - 1

#         temperature_gauge.set_value(random_weather['temperature'])
#         windspeed_gauge.set_value(random_weather['windspeed'])

#         wind_direction_series.add_points([
#             {'axis': direction, 'value': value}
#             for direction, value in zip(wind_directions, random_weather['wind_direction'])
#         ])

#         print(f"Predicted Sigheight: {predicted_sigheight}")

#         grid = np.full((1, 100), predicted_sigheight) 
#         surface.add_values(grid.tolist())  

#         time.sleep(2)

# dashboard.open(live=True)
# update_dashboard()















import pandas as pd
import numpy as np
import lightningchart as lc
import random
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Set your LightningChart license key
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load your dataset
file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

# Prepare the dataset
X = data[['temperature', 'windspeed']]
y = data['sigheight']

# Encode wind directions as one-hot
wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
encoder = OneHotEncoder(sparse_output=False)
random_wind_directions = np.random.choice(wind_directions, size=len(data))
encoded_wind_directions = pd.DataFrame(
    encoder.fit_transform(random_wind_directions.reshape(-1, 1)),
    columns=encoder.get_feature_names_out(['wind_direction'])
)

X = pd.concat([X, encoded_wind_directions], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Create the dashboard
dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

# Temperature gauge
temperature_gauge = dashboard.GaugeChart(row_index=0, column_index=0)
temperature_gauge.set_title('Temperature')
temperature_gauge.set_angle_interval(start=225, end=-45)
temperature_gauge.set_interval(start=0, end=50)
temperature_gauge.set_unit_label('Celsius')
temperature_gauge.set_unit_label_font(size=25, weight='bold')
temperature_gauge.set_value_indicators([
    {'start': 0, 'end': 10, 'color': lc.Color(0, 0, 255)},  # Blue
    {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 255)},  # Cyan
    {'start': 20, 'end': 30, 'color': lc.Color(0, 255, 0)},  # Green
    {'start': 30, 'end': 40, 'color': lc.Color(255, 255, 0)},  # Yellow
    {'start': 40, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
])
temperature_gauge.set_bar_thickness(30)
temperature_gauge.set_value_indicator_thickness(8)

# Wind direction spider chart
spider_chart = dashboard.SpiderChart(row_index=0, column_index=1)
spider_chart.set_title('Wind Direction')
wind_direction_series = spider_chart.add_series()

# Windspeed gauge
windspeed_gauge = dashboard.GaugeChart(row_index=0, column_index=2)
windspeed_gauge.set_title('Windspeed')
windspeed_gauge.set_angle_interval(start=225, end=-45)
windspeed_gauge.set_interval(start=0, end=50)
windspeed_gauge.set_unit_label('m/s')
windspeed_gauge.set_unit_label_font(size=25, weight='bold')
windspeed_gauge.set_value_indicators([
    {'start': 0, 'end': 10, 'color': lc.Color(0, 0, 255)},  # Blue
    {'start': 10, 'end': 20, 'color': lc.Color(0, 255, 255)},  # Cyan
    {'start': 20, 'end': 30, 'color': lc.Color(0, 255, 0)},  # Green
    {'start': 30, 'end': 40, 'color': lc.Color(255, 255, 0)},  # Yellow
    {'start': 40, 'end': 50, 'color': lc.Color(255, 0, 0)}  # Red
])
windspeed_gauge.set_bar_thickness(30)
windspeed_gauge.set_value_indicator_thickness(8)

# 3D surface chart for sigheight
sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
sigheight_chart.set_title('Predicted Sigheight Over Time')
sigheight_chart.set_bounding_box(x=9.0, y=1.0, z=1)
axis = sigheight_chart.get_default_x_axis()
axis.set_scroll_strategy('fitting')
axis.set_tick_strategy('Numeric')

surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=100, scroll_dimension='columns')

# Set initial palette colors for sigheight
surface.set_palette_colors(
    steps=[
        {'value': 0, 'color': lc.Color(0, 0, 255)},  # Blue for low wave height
        {'value': 0.5, 'color': lc.Color(0, 255, 255)},  # Light Blue for moderate wave height
        {'value': 1, 'color': lc.Color(255, 255, 255)},  # White for higher wave height
        {'value': 1.5, 'color': lc.Color(255, 255, 0)},  # Yellow for even higher wave height
        {'value': 2, 'color': lc.Color(255, 0, 0)},  # Red for the maximum wave height
    ],
    percentage_values=True  # Use percentage scaling for height range
)
surface.set_wireframe_stroke(1, color=lc.Color(0, 128, 255))

# Generate random weather data
def generate_random_weather_data():
    wind_direction_values = [0] * len(wind_directions)
    selected_direction = random.randint(0, len(wind_directions) - 1)
    wind_direction_values[selected_direction] = random.uniform(0, 100)

    return {
        'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
        'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
        'wind_direction': wind_direction_values
    }

# Function to update the dashboard dynamically
def update_dashboard():
    time_values = []
    start_time = time.time()
    i = 0
    x = 0
    
    while i < 1000:  
        random_weather = generate_random_weather_data()
        random_weather_df = pd.DataFrame([random_weather])

        wind_direction_encoded = pd.DataFrame([random_weather['wind_direction']], columns=encoder.get_feature_names_out(['wind_direction']))
        random_weather_df = pd.concat([random_weather_df[['temperature', 'windspeed']], wind_direction_encoded], axis=1)

        predicted_sigheight = model.predict(random_weather_df)[0]

        current_time = time.time() - start_time
        time_values.append(current_time)
        i += 1
        x = x + (random.random() * 2) - 1

        # Update temperature and windspeed gauges
        temperature_gauge.set_value(random_weather['temperature'])
        windspeed_gauge.set_value(random_weather['windspeed'])

        # Update wind direction series
        wind_direction_series.add_points([
            {'axis': direction, 'value': value}
            for direction, value in zip(wind_directions, random_weather['wind_direction'])
        ])

        # Print predicted sigheight and update 3D surface
        print(f"Predicted Sigheight: {predicted_sigheight}")

        grid = np.full((1, 100), predicted_sigheight)
        surface.add_values(grid.tolist())  # Update the height map with predicted sigheight values

        time.sleep(2)

# Open the dashboard in live mode
dashboard.open(live=True)
update_dashboard()
