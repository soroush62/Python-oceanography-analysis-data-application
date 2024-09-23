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
#     method='nearest'
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











# import pandas as pd
# import lightningchart as lc
# import numpy as np
# from scipy.interpolate import LinearNDInterpolator

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

# print(f'min z is {z.min()}')
# print(f'max z is {z.max()}')

# points = np.column_stack([x, y])
# interpolator = LinearNDInterpolator(points, z)

# grid_lon = np.linspace(x.min(), x.max(), 100)
# grid_lat = np.linspace(y.min(), y.max(), 100)
# grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# grid_wave_height = interpolator(grid_lon, grid_lat)

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

# surface_series.set_start(x=grid_lon.min(), z=grid_lat.min())
# surface_series.set_end(x=grid_lon.max(), z=grid_lat.max())

# surface_series.set_step(
#     x=(grid_lon.max() - grid_lon.min()) / grid_lon.shape[1],
#     z=(grid_lat.max() - grid_lat.min()) / grid_lat.shape[0]
# )

# surface_series.invalidate_height_map(grid_wave_height.tolist())

# surface_series.set_palette_colors(
#     steps=[
#         {"value": np.nanmin(grid_wave_height), "color": lc.Color(0, 0, 255)},       # Blue 
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





import numpy as np
import pandas as pd
import lightningchart as lc

# Set up the license
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)


beach_path = 'Dataset/beach.csv'
hour_forecast_path = 'Dataset/hour_forecast.csv'
spot_path = 'Dataset/spot.csv'
day_forecast_path = 'Dataset/day_forecast.csv'

# Reading CSV files
beach = pd.read_csv(beach_path)
hour_forecast = pd.read_csv(hour_forecast_path)
spot = pd.read_csv(spot_path)
day_forecast = pd.read_csv(day_forecast_path)

# Merge spot with beach to get latitude and longitude
merged_data = pd.merge(spot, beach[['idbeach', 'latitude', 'longitude']], on='idbeach')

# Merge day_forecast with hour_forecast to get sigheight and idbeach
forecast_data = pd.merge(hour_forecast, day_forecast[['iddayforecast', 'idbeach']], on='iddayforecast')

# Merge with the previous data
final_data = pd.merge(merged_data, forecast_data[['iddayforecast', 'sigheight', 'idbeach']], on='idbeach')

# Clean data: remove any NaN or zeros in sigheight
final_data = final_data.dropna(subset=['sigheight'])

# Prepare data for the 3D surface plot
latitude = final_data['latitude'].values
longitude = final_data['longitude'].values
sigheight = final_data['sigheight'].values

print(final_data[['latitude', 'longitude', 'sigheight']].head())
print(final_data['sigheight'].describe())  # This should give you an idea of the data range


# Get grid size for latitudes and longitudes
unique_latitudes = np.unique(latitude)
unique_longitudes = np.unique(longitude)

grid_size_x = len(unique_latitudes)
grid_size_y = len(unique_longitudes)

# Reshape sigheight into a grid
sigheight_grid = np.zeros((grid_size_x, grid_size_y))

print('SigHeight Grid Max:', np.max(sigheight_grid))
print('SigHeight Grid Min:', np.min(sigheight_grid))
print('SigHeight Grid:', sigheight_grid)


for i, lat in enumerate(unique_latitudes):
    for j, lon in enumerate(unique_longitudes):
        mask = (final_data['latitude'] == lat) & (final_data['longitude'] == lon)
        if mask.any():
            sigheight_grid[i, j] = final_data.loc[mask, 'sigheight'].mean()  # Averaging sigheight for the same lat/lon

# Create 3D chart
chart = lc.Chart3D(
    theme=lc.Themes.Dark,
    title='3D Surface Plot of SigHeight'
)

# Add surface grid series
surface_series = chart.add_surface_grid_series(
    columns=grid_size_x,
    rows=grid_size_y,
)

# Set axis ranges
surface_series.set_start(x=min(unique_latitudes), z=min(unique_longitudes))
surface_series.set_end(x=max(unique_latitudes), z=max(unique_longitudes))
surface_series.set_step(x=1, z=1)

# Set intensity interpolation
surface_series.set_intensity_interpolation(True)

# Set the height (sigheight) data
surface_series.invalidate_height_map(sigheight_grid.tolist())

surface_series.set_palette_colors(
    steps=[
        {"value": 0, "color": lc.Color(0, 0, 255)},       # Blue 
        {"value": 0.25, "color": lc.Color(0, 255, 255)},  # Cyan 
        {"value": 0.5, "color": lc.Color(0, 255, 0)},     # Green 
        {"value": 1.0, "color": lc.Color(255, 255, 0)},   # Yellow 
        {"value": 2, "color": lc.Color(255, 0, 0)}      # Red 
    ],
    look_up_property='value',
    percentage_values=False
)


# Hide the wireframe for a cleaner view
surface_series.hide_wireframe()

# Set axis titles
chart.get_default_x_axis().set_title('Latitude')
chart.get_default_y_axis().set_title('SigHeight')
chart.get_default_z_axis().set_title('Longitude')

# Add a legend
chart.add_legend(data=surface_series)

# Open the chart
chart.open()
