import numpy as np
import pandas as pd
import lightningchart as lc

lc.set_license(open('../license-key').read())

day_forecast_path = 'Dataset/day_forecast.csv'
hour_forecast_path = 'Dataset/hour_forecast.csv'

day_forecast = pd.read_csv(day_forecast_path)
hour_forecast = pd.read_csv(hour_forecast_path)

merged_data = pd.merge(day_forecast, hour_forecast[['iddayforecast', 'sigheight']], on='iddayforecast')

moon_phase_data = merged_data.groupby('moon_phase').agg({
    'moon_illumination': 'mean',
    'sigheight': 'mean'
}).reset_index()

moon_phases = moon_phase_data['moon_phase'].tolist()
illumination_values = moon_phase_data['moon_illumination'].tolist()
wave_height_values = moon_phase_data['sigheight'].tolist()

min_wave = min(wave_height_values)
max_wave = max(wave_height_values)
scaled_wave_height_values = [(value - min_wave) / (max_wave - min_wave) * 20 for value in wave_height_values]

chart = lc.SpiderChart(
    theme=lc.Themes.White,
    title='Moon Phase vs Wave Height and Moon Illumination'
)
chart.set_axis_label_font(weight='bold',size=15)
chart.set_nib_style(thickness=5, color=lc.Color(0, 0, 0))

for moon_phase in moon_phases:
    chart.add_axis(moon_phase)

series_illumination = chart.add_series()
series_illumination.set_name('Moon Illumination')
series_illumination.add_points([
    {'axis': moon_phase, 'value': value} for moon_phase, value in zip(moon_phases, illumination_values)
])


series_wave_height = chart.add_series()
series_wave_height.set_name('Wave Height (Scaled)')
series_wave_height.add_points([
    {'axis': moon_phase, 'value': value} for moon_phase, value in zip(moon_phases, scaled_wave_height_values)
])

legend = chart.add_legend()
legend.add(data=series_illumination).add(data=series_wave_height)

chart.open()

