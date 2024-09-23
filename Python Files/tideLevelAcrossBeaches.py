import pandas as pd
import numpy as np
import lightningchart as lc

lc.set_license(open('../license-key').read())

tide_path = 'Dataset/tide.csv'
beach_path = 'Dataset/beach.csv'
day_forecast_path = 'Dataset/day_forecast.csv'

tide_data = pd.read_csv(tide_path)
beach_data = pd.read_csv(beach_path)
day_forecast_data = pd.read_csv(day_forecast_path)

merged_tide = pd.merge(tide_data, day_forecast_data, on='iddayforecast')
merged_tide = pd.merge(merged_tide, beach_data, on='idbeach')

merged_tide['time'] = pd.to_datetime(merged_tide['time'])

pivot_tide = merged_tide.pivot_table(index='time', columns='name', values='height', aggfunc='mean')
pivot_tide.fillna(method='ffill', inplace=True)

beaches = pivot_tide.columns
time_values = pivot_tide.index.astype(np.int64) // 10**9  

chart = lc.ChartXY(theme=lc.Themes.White, title='Tide Levels Over Time Across Beaches')
chart.set_title('Tide Levels Over Time Across Beaches')

base_area = np.zeros(len(time_values))

for beach in beaches:
    tide_heights = pivot_tide[beach].fillna(0).values

    series = chart.add_area_series(
        data_pattern='ProgressiveX',
    )
    series.set_name(beach)
    cumulative_heights = base_area + tide_heights
    series.add(time_values.tolist(), cumulative_heights.tolist())

    base_area = cumulative_heights

chart.get_default_x_axis().set_title('Time').set_tick_strategy('DateTime')
chart.get_default_y_axis().set_title('Tide Height (m)')


legend = chart.add_legend(data=chart)

chart.open()



