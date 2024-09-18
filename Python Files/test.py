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





import pandas as pd
import lightningchart as lc
import numpy as np

# Load your license key
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the tide data
tide_path = 'Dataset/tide.csv'
tide_data = pd.read_csv(tide_path)

beach_path = 'Dataset/beach.csv'
beach_data = pd.read_csv(beach_path)

day_forecast_path = 'Dataset/day_forecast.csv'
day_forecast_data = pd.read_csv(day_forecast_path)

# Merge tide data with beach and day_forecast to get beach names and dates
merged_tide = pd.merge(tide_data, day_forecast_data, on='iddayforecast')
merged_tide = pd.merge(merged_tide, beach_data, on='idbeach')
print(merged_tide)
# Convert the 'time' column to datetime for better plotting
merged_tide['time'] = pd.to_datetime(merged_tide['time'])

# Create a pivot table for tide height over time for each beach
pivot_tide = merged_tide.pivot_table(
    index='time', columns='name', values='height', aggfunc='mean'
)
print(pivot_tide)
# Fill any missing values
pivot_tide.ffill(inplace=True)

# Prepare data for StackedAreaChart
stacked_area_data = [pivot_tide[col].values.tolist() for col in pivot_tide.columns]
time_timestamps = pivot_tide.index.astype(np.int64) // 10**9  # Convert to Unix timestamps (seconds)

# Create the stacked area chart
chart = lc.StackedAreaChart(
    theme=lc.Themes.Dark,
    title='Tide Levels Over Time Across Beaches',
    xlabel='Time',
    ylabel='Tide Height (m)'
)

# Set up the X-axis for timestamps
x_axis = chart.get_default_x_axis()
x_axis.set_interval(min(time_timestamps), max(time_timestamps))

# Add each beach's tide data as a stacked area
for beach_data in stacked_area_data:
    chart.add([beach_data])

# Customize colors for each beach
beach_colors = [
    lc.Color('blue'), lc.Color('green'), lc.Color('orange'),
    lc.Color('red'), lc.Color('purple'), lc.Color('pink'),
    lc.Color('yellow')
]

# # Open the chart in live mode
chart.open()
