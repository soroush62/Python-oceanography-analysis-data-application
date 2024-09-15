import pandas as pd
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = '../Dataset/hour_forecast.csv'
hour_forecast_data = pd.read_csv(file_path)


grouped_hour_data = hour_forecast_data.groupby('time').agg({
    'windspeed': 'mean',
    'sigheight': 'mean'
}).reset_index()

grouped_hour_data['time_in_milliseconds'] = grouped_hour_data['time']/100*3600*1000

chart = lc.ChartXY(
    theme=lc.Themes.White,
    title='Wind Speed and Wave Height Over Time'
)

wind_speed_series = chart.add_line_series()
wind_speed_series.set_name('Wind Speed')
wind_speed_series.set_line_color(lc.Color(0, 0, 255))  # Blue
wind_speed_series.set_line_thickness(2)

wave_height_series = chart.add_line_series()
wave_height_series.set_name('Wave Height')
wave_height_series.set_line_color(lc.Color(0, 255, 0))  # Green
wave_height_series.set_line_thickness(2)

wind_speed_series.append_samples(
    x_values=grouped_hour_data['time_in_milliseconds'].tolist(),
    y_values=grouped_hour_data['windspeed'].tolist()
)

wave_height_series.append_samples(
    x_values=grouped_hour_data['time_in_milliseconds'].tolist(),
    y_values=grouped_hour_data['sigheight'].tolist()
)

x_axis = chart.get_default_x_axis()
x_axis.set_title('Time')
x_axis.set_tick_strategy(strategy="Time") 

y_axis = chart.get_default_y_axis()
y_axis.set_title('Speed/Height')

chart.open()
