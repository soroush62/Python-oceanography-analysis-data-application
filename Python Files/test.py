# import numpy as np
# import pandas as pd
# import lightningchart as lc
# from scipy.interpolate import griddata

# # Load your license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load hour_forecast data
# hour_forecast_path = 'Dataset/hour_forecast.csv'
# hour_forecast = pd.read_csv(hour_forecast_path)

# # Load day_forecast data
# day_forecast_path = 'Dataset/day_forecast.csv'
# day_forecast = pd.read_csv(day_forecast_path)

# # Load beach data for latitude and longitude
# beach_path = 'Dataset/beach.csv'
# beach = pd.read_csv(beach_path)

# # First join hour_forecast with day_forecast on iddayforecast
# merged_data = pd.merge(hour_forecast, day_forecast, on='iddayforecast')

# # Now join the result with beach data on idbeach (from day_forecast)
# merged_data = pd.merge(merged_data, beach, on='idbeach')

# # Convert time values (e.g., 0, 100, 200, etc.) into a proper datetime format
# def convert_time_to_milliseconds(time_value):
#     time_str = str(int(time_value)).zfill(4)  # Convert time to a string and pad with zeros if necessary
#     return pd.to_datetime(time_str, format='%H%M')

# # Apply the conversion to the 'time' column
# merged_data['time'] = merged_data['time'].apply(convert_time_to_milliseconds)

# # Filter the necessary columns: latitude, longitude, windspeed, sigheight, and time
# data = merged_data[['latitude', 'longitude', 'windspeed', 'sigheight', 'time']]

# # Create a sorted set of unique times (sorted by time)
# unique_times = sorted(data['time'].unique())

# # Use numpy to create grid points for the surface
# latitudes = np.unique(data['latitude'])
# longitudes = np.unique(data['longitude'])

# # Create a meshgrid for latitude and longitude
# lat_grid, lon_grid = np.meshgrid(latitudes, longitudes)

# # Initialize empty grids for windspeed and wave height
# wind_grid = np.zeros((lat_grid.shape[0], lon_grid.shape[1], len(unique_times)))
# wave_height_grid = np.zeros((lat_grid.shape[0], lon_grid.shape[1], len(unique_times)))

# # For each time point, interpolate windspeed and sigheight over the grid
# for i, time in enumerate(unique_times):
#     # Subset the data for the current time point
#     current_data = data[data['time'] == time]
    
#     # Interpolate the windspeed over the latitude and longitude grid
#     wind_interpolated = griddata(
#         (current_data['latitude'], current_data['longitude']),
#         current_data['windspeed'],
#         (lat_grid, lon_grid),
#         method='linear'
#     )
    
#     # Interpolate wave height (sigheight) over the latitude and longitude grid
#     wave_height_interpolated = griddata(
#         (current_data['latitude'], current_data['longitude']),
#         current_data['sigheight'],
#         (lat_grid, lon_grid),
#         method='linear'
#     )
    
#     # Handle cases where interpolation fails (fill NaNs with nearest valid value)
#     wind_grid[:, :, i] = np.nan_to_num(wind_interpolated, nan=np.nanmean(current_data['windspeed']))
#     wave_height_grid[:, :, i] = np.nan_to_num(wave_height_interpolated, nan=np.nanmean(current_data['sigheight']))

# # Initialize a 3D chart
# chart = lc.Chart3D(
#     theme=lc.Themes.Dark,
#     title='3D Surface Plot: Windspeed and Wave Height'
# )

# # Create a SurfaceGridSeries
# surface_series = chart.add_surface_grid_series(
#     columns=len(longitudes),
#     rows=len(latitudes)
# )

# # Set start and end coordinates for the grid
# surface_series.set_start(x=0, z=0)  # Start at the first point of the grid
# surface_series.set_end(x=len(longitudes), z=len(latitudes))  # End at the last point

# # Set step size for the grid
# surface_series.set_step(
#     x=(len(longitudes)) / len(longitudes),
#     z=(len(latitudes)) / len(latitudes)
# )

# # Invalidate height map using wave_height_grid for visualization (use data at the first time step for simplicity)
# surface_series.invalidate_height_map(wave_height_grid[:, :, 0].tolist())  # You can choose other time indices if needed

# # Set color intensity based on windspeed (use data at the first time step for simplicity)
# surface_series.invalidate_intensity_values(wind_grid[:, :, 0].tolist())

# # Hide the wireframe to make it look cleaner
# surface_series.hide_wireframe()

# # Define a custom color palette for windspeed (for intensity mapping)
# surface_series.set_palette_colors(
#     steps=[
#         {"value": np.nanmin(wind_grid), "color": lc.Color(0, 0, 255)},       # Blue for lower values
#         {"value": np.nanpercentile(wind_grid, 25), "color": lc.Color(0, 255, 255)},  # Cyan for mid values
#         {"value": np.nanmedian(wind_grid), "color": lc.Color(0, 255, 0)},    # Green for median values
#         {"value": np.nanpercentile(wind_grid, 75), "color": lc.Color(255, 255, 0)},  # Yellow for upper mid values
#         {"value": np.nanmax(wind_grid), "color": lc.Color(255, 0, 0)}        # Red for higher values
#     ],
#     look_up_property='value',
#     percentage_values=False
# )

# # Set axis titles
# chart.get_default_x_axis().set_title('Longitude')
# chart.get_default_y_axis().set_title('Wave Height (Sigheight)')
# chart.get_default_z_axis().set_title('Latitude')

# # Add legend
# chart.add_legend(data=surface_series)

# # Open the chart
# chart.open()






# import pandas as pd
# import numpy as np
# import lightningchart as lc

# # Load your license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable2.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load tide data and beach data
# tide_path = 'Dataset/tide.csv'
# tide_data = pd.read_csv(tide_path)

# beach_path = 'Dataset/beach.csv'
# beach_data = pd.read_csv(beach_path)

# day_forecast_path = 'Dataset/day_forecast.csv'
# day_forecast_data = pd.read_csv(day_forecast_path)

# # Merge tide data with beach and day_forecast to get beach names and dates
# merged_tide = pd.merge(tide_data, day_forecast_data, on='iddayforecast')
# merged_tide = pd.merge(merged_tide, beach_data, on='idbeach')

# # Convert the 'time' column to datetime without specifying format (automatic inference)
# merged_tide['time'] = pd.to_datetime(merged_tide['time'], errors='coerce')

# # Create a pivot table for tide height over time for each beach
# pivot_tide = merged_tide.pivot_table(
#     index='time', columns='name', values='height', aggfunc='mean'
# )

# # Fill missing values with forward fill to avoid gaps in the data
# pivot_tide.ffill(inplace=True)

# # Prepare data in the format that LightningChart's StackedAreaChart expects
# stacked_area_data = pivot_tide.values.tolist()

# # Create the stacked area chart
# chart = lc.StackedAreaChart(
#     data=stacked_area_data,
#     theme=lc.Themes.Dark,  # Choose a theme for the chart
#     title='Tide Levels Over Time Across Beaches',
#     xlabel='Time',
#     ylabel='Tide Height (m)'
# )

# # Set the x-axis with datetime (convert to strings if needed)
# chart.get_default_x_axis().set_interval(start=0, end=len(pivot_tide.index))

# # Open the chart in live mode
# chart.open(live=True)





import numpy as np
import pandas as pd
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

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
