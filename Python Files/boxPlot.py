import lightningchart as lc
import pandas as pd
import numpy as np

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight']
box_plot_data = [data[column].tolist() for column in columns]

color_map = {
    'temperature': lc.Color(255, 235, 205),  # Light Coral
    'windspeed': lc.Color(240, 230, 140),  # Khaki
    'sigheight': lc.Color(255, 218, 185),  # Peach Puff
    'humidity': lc.Color(255, 239, 213),  # Papaya Whip
    'feelslike': lc.Color(255, 250, 205),  # Lemon Chiffon
    'swellheight': lc.Color(255, 245, 238)  # Seashell
}

dashboard = lc.Dashboard(
    rows=2,
    columns=3,
    theme=lc.Themes.Dark
)

def add_box_plot_to_chart(chart, column_data, column_name, column_index):
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    median = np.median(column_data)
    min_val = np.min(column_data)
    max_val = np.max(column_data)
    
    series = chart.add_box_series()
    series.add(
        start=column_index - 0.4,
        end=column_index + 0.4,
        median=float(median),
        lower_quartile=float(q1),
        upper_quartile=float(q3),
        lower_extreme=float(min_val),
        upper_extreme=float(max_val)
    )
    series.set_name(column_name)

for i, (label, values) in enumerate(zip(columns, box_plot_data)):
    row_index = i // 3
    col_index = i % 3
    chart = dashboard.ChartXY(
        column_index=col_index,
        row_index=row_index,
        title=label
    )
    add_box_plot_to_chart(chart, values, label, i)
    chart.set_series_background_color(color_map[label])
    x_axis = chart.get_default_x_axis()
    x_axis.set_title('Category')
    y_axis = chart.get_default_y_axis()
    y_axis.set_title('Values')

dashboard.open()
