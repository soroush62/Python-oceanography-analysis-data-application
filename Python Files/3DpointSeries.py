# import pandas as pd
# import numpy as np
# import lightningchart as lc

# # Load your shared variable key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load datasets (assuming the file paths are correct)
# day_forecast_path = 'Dataset/day_forecast.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# beach_path = 'Dataset/beach.csv'

# day_forecast = pd.read_csv(day_forecast_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# beach = pd.read_csv(beach_path)

# # Merge datasets based on common fields 'iddayforecast' and 'idbeach'
# merged_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# merged_data = pd.merge(merged_data, beach[['idbeach', 'name']], on='idbeach')

# # Map beach names to numerical values for Z-axis
# beach_mapping = {name: idx for idx, name in enumerate(merged_data['name'].unique())}
# merged_data['beach_num'] = merged_data['name'].map(beach_mapping)

# # Prepare data for plotting
# x_values = merged_data['windspeed'].astype(float).to_numpy()  # Windspeed as X-axis
# y_values = merged_data['cloundover'].astype(float).to_numpy()  # Cloud cover as Y-axis
# z_values = merged_data['beach_num'].astype(float).to_numpy()  # Beach (numerical) as Z-axis
# color_values = merged_data['time'].astype(float).to_numpy()  # Time or Tide as color values

# # Create a 3D chart
# chart = lc.Chart3D(
#     title='3D Scatter Plot: Cloud Cover vs Windspeed vs Beach',
#     theme=lc.Themes.Dark
# )

# # Add a point series
# scatter_series = chart.add_point_series(individual_lookup_values_enabled=True)

# # Set point appearance
# scatter_series.set_point_shape('sphere')
# scatter_series.set_point_size(6.0)  # Increase point size for better visibility

# # Map the color of the points to the 'time' values using a color palette
# scatter_series.set_palette_point_colors(
#     steps=[
#         {"value": min(color_values), "color": lc.Color(0, 0, 255)},  # Blue for low time values
#         {"value": max(color_values), "color": lc.Color(255, 0, 0)}  # Red for high time values
#     ],
#     look_up_property='value',  # Use individual values for color lookup
#     interpolate=True
# )

# # Add data to the point series and include the color values as the lookup property
# scatter_series.add(x_values, y_values, z_values, lookup_values=color_values)

# # Set axis titles
# chart.get_default_x_axis().set_title('Windspeed (m/s)')
# chart.get_default_y_axis().set_title('Cloud Cover (%)')
# chart.get_default_z_axis().set_title('Beach Index')

# # Add a custom legend for the beach names and their corresponding numerical values
# legend = chart.add_legend()
# # for beach, idx in beach_mapping.items():
# #     legend.add(f"{beach}")

# # Open the chart
# chart.open()












# import lightningchart as lc
# import pandas as pd
# import numpy as np

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# day_forecast_path = 'Dataset/day_forecast.csv'
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# beach_path = 'Dataset/beach.csv'

# day_forecast = pd.read_csv(day_forecast_path)
# hour_forecast = pd.read_csv(hour_forecast_path)
# beach = pd.read_csv(beach_path)

# merged_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
# merged_data = pd.merge(merged_data, beach[['idbeach', 'name']], on='idbeach')

# beach_mapping = {name: idx for idx, name in enumerate(merged_data['name'].unique())}
# merged_data['beach_num'] = merged_data['name'].map(beach_mapping)

# chart = lc.Chart3D(
#     title='3D Scatter Plot: Cloud Cover vs Windspeed vs Beach',
#     theme=lc.Themes.Dark
# )

# x_values = merged_data['windspeed'].astype(float).to_numpy()  # Windspeed as X-axis
# y_values = merged_data['cloundover'].astype(float).to_numpy()  # Cloud cover as Y-axis
# z_values = merged_data['beach_num'].astype(float).to_numpy()  # Beach (numerical) as Z-axis
# lookup_values  = merged_data['time'].astype(float).to_numpy()  # Time or Tide as color values
# # print(color_values)
# scatter_series = chart.add_point_series(
#     individual_lookup_values_enabled=True,
#     individual_point_size_axis_enabled=True,
#     individual_point_size_enabled=True)

# scatter_series.set_point_shape('sphere')
# scatter_series.set_point_size(4.0)

# scatter_series.set_palette_point_colors(
#     steps=[
#         {"value": min(lookup_values ), "color": lc.Color(0, 0, 255)},  # Blue for low values
#         {"value": max(lookup_values ), "color": lc.Color(255, 0, 0)}  # Red for high values
#     ],
#     look_up_property='value',
#     percentage_values=True, 
#     interpolate=True
# )

# scatter_series.add(x_values, y_values, z_values)

# chart.get_default_x_axis().set_title('DAILY_YIELD')
# chart.get_default_y_axis().set_title('AMBIENT_TEMPERATURE')
# chart.get_default_z_axis().set_title('MODULE_TEMPERATURE')

# chart.open()





import lightningchart as lc
import pandas as pd
import numpy as np

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

day_forecast_path = 'Dataset/day_forecast.csv'
hour_forecast_path = 'Dataset/hour_forecast.csv'
beach_path = 'Dataset/beach.csv'

day_forecast = pd.read_csv(day_forecast_path)
hour_forecast = pd.read_csv(hour_forecast_path)
beach = pd.read_csv(beach_path)

merged_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')
merged_data = pd.merge(merged_data, beach[['idbeach', 'name']], on='idbeach')

beach_mapping = {name: idx for idx, name in enumerate(merged_data['name'].unique())}
merged_data['beach_num'] = merged_data['name'].map(beach_mapping)

chart = lc.Chart3D(
    title='3D Scatter Plot: Cloud Cover vs Windspeed vs Beach',
    theme=lc.Themes.Dark
)

x_values = merged_data['windspeed'].astype(float).to_numpy()  
y_values = merged_data['cloundover'].astype(float).to_numpy()  
z_values = merged_data['beach_num'].astype(float).to_numpy()  

lookup_values = merged_data['time'].astype(float).to_numpy()  
min_lookup = np.min(lookup_values)
max_lookup = np.max(lookup_values)
normalized_lookup_values = (lookup_values - min_lookup) / (max_lookup - min_lookup)  

scatter_series = chart.add_point_series(
    individual_lookup_values_enabled=True,
    individual_point_size_axis_enabled=True,
    individual_point_size_enabled=True
)

scatter_series.set_point_shape('sphere')
scatter_series.set_point_size(8.0)

scatter_series.set_palette_point_colors(
    steps=[
        {"value": 0.0, "color": lc.Color(0, 0, 255)}, 
        {"value": 1.0, "color": lc.Color(255, 0, 0)}   
    ],
    look_up_property='value',  
    percentage_values=True,    
    interpolate=True           
)

data = [
    {
        'x': x_values[i],  
        'y': y_values[i],  
        'z': z_values[i],  
        'value': normalized_lookup_values[i]  
    }
    for i in range(len(x_values))
]

scatter_series.add(data)

x_axis=chart.get_default_x_axis().set_title('Windspeed')
chart.get_default_y_axis().set_title('Cloud Cover')
chart.get_default_z_axis().set_title('Beach (numerical)')
x_axis.set_tick_strategy('Empty')
chart.add_legend(data=scatter_series, title='Time')

chart.open()
