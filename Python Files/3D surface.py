# import pandas as pd
# import lightningchart as lc
# import numpy as np
# from scipy.interpolate import griddata

# # Set up the license
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

# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach']], on='idbeach')

# x = final_data['longitude'].values
# y = final_data['latitude'].values
# z = final_data['sigheight'].values

# grid_lat, grid_lon = np.meshgrid(
#     np.linspace(x.min(), x.max(), 100),
#     np.linspace(y.min(), y.max(), 100)
# )

# grid_wave_height = griddata(
#     (x, y), z,
#     (grid_lat, grid_lon),
#     method='cubic'
# )

# nan_mask = np.isnan(grid_wave_height)
# grid_wave_height[nan_mask] = np.nanmean(z)

# chart = lc.Chart3D(
#     theme=lc.Themes.Dark,
#     title='3D Seabed Visualization: Wave Heights by Spot'
# )

# surface_series = chart.add_surface_grid_series(
#     columns=grid_wave_height.shape[1],
#     rows=grid_wave_height.shape[0]
# )

# surface_series.set_start(x=x.min(), z=y.min())
# surface_series.set_end(x=x.max(), z=y.max())
# print(grid_wave_height[:1])
# print(grid_wave_height.shape[1])
# surface_series.set_step(
#     x=(x.max() - x.min()) / grid_wave_height.shape[1],
#     z=(y.max() - y.min()) / grid_wave_height.shape[0]
# )

# surface_series.invalidate_height_map(grid_wave_height.tolist())

# surface_series.set_palette_colors(
#     steps=[
#         {"value": np.nanmin(grid_wave_height), "color": lc.Color(0, 0, 255)},       # Blue 
#         {"value": np.nanpercentile(grid_wave_height, 25), "color": lc.Color(0, 255, 255)},  # Cyan 
#         {"value": np.nanmedian(grid_wave_height), "color": lc.Color(0, 255, 0)},    # Green 
#         {"value": np.nanpercentile(grid_wave_height, 75), "color": lc.Color(255, 255, 0)},  # Yellow 
#         {"value": np.nanmax(grid_wave_height), "color": lc.Color(255, 0, 0)}        # Red 
#     ],
#     look_up_property='value',
#     percentage_values=False
# )

# surface_series.invalidate_intensity_values(grid_wave_height.tolist())

# surface_series.hide_wireframe()

# chart.get_default_x_axis().set_title('Longitude')
# chart.get_default_y_axis().set_title('Wave Height (m)')
# chart.get_default_z_axis().set_title('Latitude')

# chart.add_legend(data=surface_series)

# chart.open()






# import pandas as pd
# import lightningchart as lc
# import numpy as np

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

# final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach']], on='idbeach')

# x_values = final_data['longitude'].values
# y_values = final_data['latitude'].values
# z_values = final_data['sigheight'].values

# unique_x = np.unique(x_values)
# unique_y = np.unique(y_values)

# data_grid = np.zeros((len(unique_y), len(unique_x)))

# for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
#     x_index = np.where(unique_x == x)[0][0]
#     y_index = np.where(unique_y == y)[0][0]
#     data_grid[y_index, x_index] = z  

# data_for_surface = data_grid.tolist()

# chart = lc.Surface3D(
#     data=data_for_surface,  # Pass the grid data directly
#     min_value=None,         # Automatically determine minimum
#     max_value=None,         # Automatically determine maximum
#     min_color=lc.Color(0, 0, 255),   # Blue for lowest points
#     max_color=lc.Color(255, 0, 0),   # Red for highest points
#     theme=lc.Themes.Dark,    # Use dark theme
#     title='3D Seabed Visualization: Wave Heights by Spot',
#     xlabel='Longitude',
#     ylabel='Latitude',
#     zlabel='Wave Height (m)'
# )

# chart.open()











import pandas as pd
import lightningchart as lc
import numpy as np
from scipy.interpolate import LinearNDInterpolator

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

final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach']], on='idbeach')

x = final_data['longitude'].values
y = final_data['latitude'].values
z = final_data['sigheight'].values

print(f'min z is {z.min()}')
print(f'max z is {z.max()}')

points = np.column_stack([x, y])
interpolator = LinearNDInterpolator(points, z)

grid_lon = np.linspace(x.min(), x.max(), 100)
grid_lat = np.linspace(y.min(), y.max(), 100)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

grid_wave_height = interpolator(grid_lon, grid_lat)

nan_mask = np.isnan(grid_wave_height)
grid_wave_height[nan_mask] = np.nanmean(z)

chart = lc.Chart3D(
    theme=lc.Themes.Dark,
    title='3D Seabed Visualization: Wave Heights by Spot'
)

surface_series = chart.add_surface_grid_series(
    columns=grid_wave_height.shape[1],
    rows=grid_wave_height.shape[0]
)

surface_series.set_start(x=grid_lon.min(), z=grid_lat.min())
surface_series.set_end(x=grid_lon.max(), z=grid_lat.max())

surface_series.set_step(
    x=(grid_lon.max() - grid_lon.min()) / grid_lon.shape[1],
    z=(grid_lat.max() - grid_lat.min()) / grid_lat.shape[0]
)

surface_series.invalidate_height_map(grid_wave_height.tolist())

surface_series.set_palette_colors(
    steps=[
        {"value": np.nanmin(grid_wave_height), "color": lc.Color(0, 0, 255)},       # Blue 
        {"value": np.nanpercentile(grid_wave_height, 75), "color": lc.Color(255, 255, 0)},  # Yellow 
        {"value": np.nanmax(grid_wave_height), "color": lc.Color(255, 0, 0)}        # Red 
    ],
    look_up_property='value',
    percentage_values=False
)

surface_series.invalidate_intensity_values(grid_wave_height.tolist())

surface_series.hide_wireframe()

chart.get_default_x_axis().set_title('Longitude')
chart.get_default_y_axis().set_title('Wave Height (m)')
chart.get_default_z_axis().set_title('Latitude')

chart.add_legend(data=surface_series)

chart.open()
