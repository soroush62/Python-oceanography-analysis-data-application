# import pandas as pd
# import lightningchart as lc
# import numpy as np

# # Load your license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load the tide data
# tide_path = 'Dataset/tide.csv'
# tide_data = pd.read_csv(tide_path)

# beach_path = 'Dataset/beach.csv'
# beach_data = pd.read_csv(beach_path)

# day_forecast_path = 'Dataset/day_forecast.csv'
# day_forecast_data = pd.read_csv(day_forecast_path)

# merged_tide = pd.merge(tide_data, day_forecast_data, on='iddayforecast')
# merged_tide = pd.merge(merged_tide, beach_data, on='idbeach')
# merged_tide['time'] = pd.to_datetime(merged_tide['time'])
# # print(merged_tide)

# data=[]
# time_values = merged_tide['time'].astype(np.int64) // 10**9
# unique_tide=merged_tide['name'].unique()
# print(unique_tide)
# for beach  in unique_tide:
#     beach_data=merged_tide[merged_tide['name']==beach]
#     beach_data=beach_data.sort_values(by='time')
#     tide_height=beach_data['height'].tolist()
#     data.append(tide_height)

# # print(data[:5])
# chart = lc.StackedAreaChart(
#     theme=lc.Themes.Dark,
#     data=data,
#     title='Tide Levels Over Time Across Beaches',
#     xlabel='Time',
#     ylabel='Tide Height (m)'
# )
# legend=chart.add_legend(data=chart)
# chart.open()






import pandas as pd
import lightningchart as lc
import numpy as np

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

tide_path = 'Dataset/tide.csv'
tide_data = pd.read_csv(tide_path)

beach_path = 'Dataset/beach.csv'
beach_data = pd.read_csv(beach_path)

day_forecast_path = 'Dataset/day_forecast.csv'
day_forecast_data = pd.read_csv(day_forecast_path)

merged_tide = pd.merge(tide_data, day_forecast_data, on='iddayforecast')
merged_tide = pd.merge(merged_tide, beach_data, on='idbeach')
merged_tide['time'] = pd.to_datetime(merged_tide['time'])

chart = lc.ChartXY(
    theme=lc.Themes.Dark,
    title='Tide Levels Over Time Across Beaches'
)

x_axis = chart.get_default_x_axis().set_title('Time')
y_axis = chart.get_default_y_axis().set_title('Tide Height (m)')
x_axis.set_tick_strategy('DateTime')
unique_beaches = merged_tide['name'].unique()

for beach in unique_beaches:
    beach_data = merged_tide[merged_tide['name'] == beach]    
    beach_data = beach_data.sort_values(by='time')
    
    time_stamps = (beach_data['time'].astype(np.int64) // 10**9).tolist()  
    tide_heights = beach_data['height'].tolist()
    
    area_series = chart.add_area_series(data_pattern='ProgressiveX')
    
    area_series.add(time_stamps, tide_heights)    
    area_series.set_name(beach)

legend = chart.add_legend(data=chart)

chart.open()
