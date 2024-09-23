import pandas as pd
import numpy as np
import lightningchart as lc
import random
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

lc.set_license(open('../license-key').read())

file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

X = data[['temperature', 'windspeed']]
y = data['sigheight']

wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
encoder = OneHotEncoder(sparse=False)
random_wind_directions = np.random.choice(wind_directions, size=len(data))
encoded_wind_directions = pd.DataFrame(
    encoder.fit_transform(random_wind_directions.reshape(-1, 1)),
    columns=encoder.get_feature_names_out(['wind_direction'])
)

X = pd.concat([X, encoded_wind_directions], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

dashboard = lc.Dashboard(theme=lc.Themes.Dark, rows=2, columns=3)

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

spider_chart = dashboard.SpiderChart(row_index=0, column_index=1)
spider_chart.set_title('Wind Direction')
wind_direction_series = spider_chart.add_series()

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

sigheight_chart = dashboard.Chart3D(row_index=1, column_index=0, column_span=3)
sigheight_chart.set_title('Predicted Sigheight Over Time')
sigheight_chart.set_bounding_box(x=9.0, y=1.0, z=1)
axis = sigheight_chart.get_default_x_axis()
axis.set_scroll_strategy('fitting')
axis.set_tick_strategy('Numeric')
surface = sigheight_chart.add_surface_scrolling_grid_series(columns=100, rows=100, scroll_dimension='columns')
surface.set_min_max_palette_colors(
    min_value=-50,
    max_value=50,
    min_color=lc.Color('#00ffff'),
    max_color=lc.Color('#ffff00'),
    look_up_property='y'
)
surface.set_wireframe_stroke(1, color=lc.Color(0, 128, 255))

def generate_random_weather_data():
    wind_direction_values = [0] * len(wind_directions)
    selected_direction = random.randint(0, len(wind_directions) - 1)
    wind_direction_values[selected_direction] = random.uniform(0, 100) 

    return {
        'temperature': np.random.uniform(X['temperature'].min(), X['temperature'].max()),
        'windspeed': np.random.uniform(X['windspeed'].min(), X['windspeed'].max()),
        'wind_direction': wind_direction_values 
    }

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

        temperature_gauge.set_value(random_weather['temperature'])
        windspeed_gauge.set_value(random_weather['windspeed'])

        wind_direction_series.add_points([
            {'axis': direction, 'value': value}
            for direction, value in zip(wind_directions, random_weather['wind_direction'])
        ])

        print(f"Predicted Sigheight: {predicted_sigheight}")

        grid = np.full((1, 100), predicted_sigheight) 
        surface.add_values(grid.tolist())  

        time.sleep(2)

dashboard.open(live=True)
update_dashboard()