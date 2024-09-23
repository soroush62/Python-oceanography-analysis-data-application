import pandas as pd
import numpy as np
import lightningchart as lc

lc.set_license(open('../license-key').read())

file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight', 'preciptation', 
           'pressure', 'cloundover', 'windchill', 'windgust', 'watertemp']

dashboard = lc.Dashboard(
    rows=3,  
    columns=4,  
    theme=lc.Themes.Dark  
)

def create_histogram(column_data, column_name, row_index, col_index):
    counts, bin_edges = np.histogram(column_data, bins=10)
    bar_data = [{'category': f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}', 'value': int(counts[i])} for i in range(len(counts))]
    chart = dashboard.BarChart(
        column_index=col_index,
        row_index=row_index,
        row_span=1,  
        column_span=1, 
    )
    chart.set_title(f'Histogram of {column_name}')
    chart.set_bars_effect(True)
    
    chart.set_data(bar_data)

    chart.set_palette_colors(
        steps=[
            {'value': 0, 'color': lc.Color('blue')}, 
            {'value': 0.5, 'color': lc.Color('yellow')}, 
            {'value': 1, 'color': lc.Color('red')}  
        ],
        percentage_values=True 
    )

for i, column in enumerate(columns):
    row_index = i // 4 
    col_index = i % 4  
    
    create_histogram(data[column].dropna(), column, row_index, col_index)

dashboard.open()