# import pandas as pd
# import numpy as np
# import lightningchart as lc
# from scipy.stats import gaussian_kde

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable2.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# hour_forecast_data = pd.read_csv('Dataset/hour_forecast.csv')
# day_forecast_data = pd.read_csv('Dataset/day_forecast.csv')

# merged_data = pd.merge(hour_forecast_data, day_forecast_data[['iddayforecast', 'idbeach']], on='iddayforecast')

# features = ['temperature', 'windspeed', 'sigheight', 'humidity', 'preciptation']

# unique_identifiers = merged_data['idbeach'].unique()

# color_map = {key: lc.Color(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for key in unique_identifiers}

# def create_density_chart(title, values_dict):
#     chart = lc.ChartXY(theme=lc.Themes.Dark)
#     chart.set_title(title)
#     chart.set_title_font(size=15)
#     chart.set_title_color(lc.Color(255, 255, 255))
#     chart.set_padding(0)
#     legend = chart.add_legend()

#     for key, values in values_dict.items():
#         values_np = np.array(values)
#         density = gaussian_kde(values_np)
#         x_vals = np.linspace(values_np.min(), values_np.max(), 100)
#         y_vals = density(x_vals)

#         series = chart.add_area_series()
#         series.add(x_vals.tolist(), y_vals.tolist())
#         series.set_name(f'Density - Beach {key}')
#         series.set_fill_color(color_map[key])
#         legend.add(data=series)

#     chart.get_default_x_axis().set_title('Value')
#     chart.get_default_y_axis().set_title('Density')
#     chart.open()

# def create_scatter_chart(title, data_dict, xlabel, ylabel):
#     chart = lc.ChartXY(theme=lc.Themes.Dark)
#     chart.set_title(title)
#     chart.set_title_font(size=15)
#     chart.set_title_color(lc.Color(255, 255, 255))
#     chart.set_padding(0)

#     for key, (x_values, y_values) in data_dict.items():
#         scatter_series = chart.add_point_series()
#         scatter_series.add(x_values, y_values)
#         scatter_series.set_point_color(color_map[key])
#         scatter_series.set_point_size(2)

#     chart.get_default_x_axis().set_title(xlabel)
#     chart.get_default_y_axis().set_title(ylabel)
#     chart.open()

# for row_index, y_col in enumerate(features):
#     for column_index, x_col in enumerate(features):
#         if row_index == column_index:
#             values_dict = {key: merged_data[merged_data['idbeach'] == key][x_col].astype(float).tolist() for key in unique_identifiers}
#             create_density_chart(f'Density of {x_col}', values_dict)
#         else:
#             data_dict = {key: (merged_data[merged_data['idbeach'] == key][x_col].astype(float).tolist(),
#                                merged_data[merged_data['idbeach'] == key][y_col].astype(float).tolist()) for key in unique_identifiers}
#             create_scatter_chart(f'{x_col} vs {y_col}', data_dict, x_col, y_col)





import pandas as pd
import numpy as np
import lightningchart as lc
from scipy.stats import gaussian_kde

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable2.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

hour_forecast_data = pd.read_csv('Dataset/hour_forecast.csv')
day_forecast_data = pd.read_csv('Dataset/day_forecast.csv')

merged_data = pd.merge(hour_forecast_data, day_forecast_data[['iddayforecast', 'idbeach']], on='iddayforecast')

features = ['temperature', 'windspeed', 'sigheight', 'humidity', 'preciptation']
unique_identifiers = merged_data['idbeach'].unique()
print(unique_identifiers)
color_map = {key: lc.Color(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for key in unique_identifiers}

dashboard = lc.Dashboard(
    rows=len(features),
    columns=len(features),
    theme=lc.Themes.Dark
)

def create_density_chart(title, values_dict, column_index, row_index):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)
    chart.set_title_font(size=15)
    chart.set_title_color(lc.Color(255, 255, 255))
    chart.set_padding(0)

    for key, values in values_dict.items():
        values_np = np.array(values)
        density = gaussian_kde(values_np)
        x_vals = np.linspace(values_np.min(), values_np.max(), 100)
        y_vals = density(x_vals)

        series = chart.add_area_series()
        series.add(x_vals.tolist(), y_vals.tolist())
        series.set_name(f'Density - Beach {key}')
        series.set_fill_color(color_map[key])

    chart.get_default_x_axis().set_title('Value')
    chart.get_default_y_axis().set_title('Density')
    

def create_scatter_chart(title, data_dict, xlabel, ylabel, column_index, row_index):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)
    chart.set_title_font(size=15)
    chart.set_title_color(lc.Color(255, 255, 255))
    chart.set_padding(0)

    for key, (x_values, y_values) in data_dict.items():
        scatter_series = chart.add_point_series()
        scatter_series.add(x_values, y_values)
        scatter_series.set_point_color(color_map[key])
        scatter_series.set_point_size(3)

    chart.get_default_x_axis().set_title(xlabel)
    chart.get_default_y_axis().set_title(ylabel)

for row_index, y_col in enumerate(features):
    for column_index, x_col in enumerate(features):
        if row_index == column_index:
            values_dict = {key: merged_data[merged_data['idbeach'] == key][x_col].astype(float).tolist() for key in unique_identifiers}
            create_density_chart(f'Density of {x_col}', values_dict, column_index, row_index)
        else:
            data_dict = {key: (
                merged_data[merged_data['idbeach'] == key][x_col].astype(float).tolist(),
                merged_data[merged_data['idbeach'] == key][y_col].astype(float).tolist()
            ) for key in unique_identifiers}
            create_scatter_chart(f'{x_col} vs {y_col}', data_dict, x_col, y_col, column_index, row_index)

dashboard.open()
