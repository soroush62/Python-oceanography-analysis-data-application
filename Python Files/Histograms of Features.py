# import pandas as pd
# import numpy as np
# import lightningchart as lc

# # Set up your LightningChart license key
# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load the dataset
# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# # Select the columns to generate histograms for
# columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight', 'preciptation', 
#            'pressure', 'cloundover', 'windchill', 'windgust', 'swellheight', 'watertemp']

# # Create a Dashboard
# dashboard = lc.Dashboard(
#     rows=3,
#     columns=4,
#     theme=lc.Themes.Dark  # You can switch themes here
# )

# # Function to create histograms
# def create_histogram(dashboard, column_data, column_name, row_index, col_index):
#     chart = dashboard.ChartXY(
#         row_index=row_index,
#         column_index=col_index,
#         title=f'Histogram of {column_name}'
#     )
    
#     # Create histogram data by binning
#     counts, bins = np.histogram(column_data, bins=30)  # Adjust the bin size accordingly
    
#     # Convert to Python built-in types (int and float)
#     counts = counts.astype(int).tolist()
#     bins = bins.astype(float).tolist()
    
#     # Add a scatter plot to simulate a bar chart for the histogram
#     series = chart.add_point_series()
#     for i in range(len(bins) - 1):
#         # Simulate bars by adding two points for each bin
#         series.add([bins[i], bins[i+1]], [counts[i], counts[i]])

#     # Customize X and Y axis labels
#     x_axis = chart.get_default_x_axis()
#     x_axis.set_title(column_name)
    
#     y_axis = chart.get_default_y_axis()
#     y_axis.set_title('Frequency')

# # Add histograms to the dashboard
# for i, column in enumerate(columns):
#     row_index = i // 4  # Calculate row index (3 rows)
#     col_index = i % 4   # Calculate column index (4 columns)
#     create_histogram(dashboard, data[column].dropna(), column, row_index, col_index)

# # Show the dashboard
# dashboard.open()





# import pandas as pd
# import numpy as np
# import lightningchart as lc

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# # Select the columns to generate histograms for
# columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight', 'preciptation', 
#            'pressure', 'cloundover', 'windchill', 'windgust', 'swellheight', 'watertemp']

# # Create a Dashboard
# dashboard = lc.Dashboard(
#     rows=3,  # Number of rows in the dashboard
#     columns=4,  # Number of columns in the dashboard
#     theme=lc.Themes.Dark  # Set theme
# )

# # Function to create a BarChart for histograms
# def create_histogram(column_data, column_name, row_index, col_index):
#     # Create a BarChart and set its position in the dashboard
#     chart = dashboard.BarChart(
#         column_index=col_index,
#         row_index=row_index,
#         row_span=1,  # Span over 1 row
#         column_span=1,  # Span over 1 column
#     )
#     chart.set_title(f'Histogram of {column_name}')

#     # Create histogram data by binning
#     counts, bins = np.histogram(column_data, bins=30)  # Adjust the bin size accordingly

#     # Prepare bar chart data in the correct format
#     categories = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins) - 1)]
#     data_for_barchart = [{"category": category, "value": int(count)} for category, count in zip(categories, counts)]

#     # Set bar chart data
#     chart.set_data(data_for_barchart)

#     # Optionally, set a custom color for the bars
#     chart.set_bars_color(lc.Color(50, 150, 255))  # Set a custom color for the bars

# # Add histograms to the dashboard for each column
# for i, column in enumerate(columns):
#     row_index = i // 4  # Calculate row index (3 rows)
#     col_index = i % 4   # Calculate column index (4 columns)
    
#     create_histogram(data[column].dropna(), column, row_index, col_index)

# dashboard.open()





import pandas as pd
import numpy as np
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable2.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

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
