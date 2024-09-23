import pandas as pd
import numpy as np
import lightningchart as lc
from scipy.interpolate import griddata

# Load the license key
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

grid_lon = np.linspace(x.min(), x.max(), 100)
grid_lat = np.linspace(y.min(), y.max(), 100)

grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

grid_wave_height = griddata((x, y), z, (grid_lon, grid_lat), method='cubic')

grid_wave_height[np.isnan(grid_wave_height)] = np.nanmean(z)

chart = lc.ChartXY(
    title='Heatmap of Wave Height by Longitude and Latitude',
    theme=lc.Themes.Dark
)

heatmap = chart.add_heatmap_grid_series(
    rows=grid_lat.shape[0],
    columns=grid_lon.shape[1]
)

heatmap.set_start(x=grid_lon.min(), y=grid_lat.min())
heatmap.set_end(x=grid_lon.max(), y=grid_lat.max())
heatmap.set_step(x=(grid_lon.max() - grid_lon.min()) / grid_lon.shape[1], y=(grid_lat.max() - grid_lat.min()) / grid_lat.shape[0])

heatmap.invalidate_intensity_values(grid_wave_height.tolist())
heatmap.hide_wireframe()

chart.get_default_x_axis().set_title('Longitude')
chart.get_default_y_axis().set_title('Latitude')

custom_palette = [
    {"value": np.nanmin(grid_wave_height), "color": lc.Color(0, 0, 255)},  # Blue
    {"value": np.nanpercentile(grid_wave_height, 25), "color": lc.Color(0, 255, 255)},  # Cyan
    {"value": np.nanmedian(grid_wave_height), "color": lc.Color(0, 255, 0)},  # Green
    {"value": np.nanpercentile(grid_wave_height, 75), "color": lc.Color(255, 255, 0)},  # Yellow
    {"value": np.nanmax(grid_wave_height), "color": lc.Color(255, 0, 0)}  # Red
]

heatmap.set_palette_colors(
    steps=custom_palette,
    look_up_property='value',
    interpolate=True
)
chart.add_legend(data=heatmap, title="Wave Height")

chart.open()
