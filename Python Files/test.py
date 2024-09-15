import pandas as pd
import numpy as np
import lightningchart as lc

# Set the license
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the dataset
file_path = 'Dataset/hour_forecast.csv'
hour_forecast_data = pd.read_csv(file_path)

numeric_columns = hour_forecast_data.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
corr_matrix = numeric_columns.corr()

# Convert the correlation matrix to a list of lists for use in the heatmap
heatmap_data = corr_matrix.values.tolist()

# Initialize the LightningChart for heatmap visualization
chart = lc.ChartXY(
    theme=lc.Themes.White,
    title='Correlation Heatmap of Sea Forecast Data'
)

# Create a heatmap grid series
series = chart.add_heatmap_grid_series(
    columns=len(heatmap_data),  # Number of features (columns)
    rows=len(heatmap_data[0])   # Number of features (rows)
)

# Set chart appearance
series.hide_wireframe()
series.set_intensity_interpolation(False)
series.invalidate_intensity_values(heatmap_data)

# Set heatmap color palette (-1.0 for blue, 0.0 for white, 1.0 for red)
series.set_palette_colors(
    steps=[
        {"value": -1.0, "color": lc.Color(0, 0, 255)},  # Blue (negative correlation)
        {"value": 0.0, "color": lc.Color(255, 255, 255)},  # White (neutral correlation)
        {"value": 1.0, "color": lc.Color(255, 0, 0)}  # Red (positive correlation)
    ],
    look_up_property='value',
    percentage_values=True
)

# Customize X and Y axis
x_axis = chart.get_default_x_axis()
x_axis.set_title('Sea Forecast Feature Index')
x_axis.set_interval(0, numeric_columns.shape[1])

y_axis = chart.get_default_y_axis()
y_axis.set_title('Sea Forecast Feature Index')
y_axis.set_interval(0, numeric_columns.shape[1])

# Open the chart
chart.open()