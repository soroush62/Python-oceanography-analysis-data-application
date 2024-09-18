# import pandas as pd
# import numpy as np
# import lightningchart as lc

# with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# file_path = 'Dataset/hour_forecast.csv'
# data = pd.read_csv(file_path)

# columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight']

# clean_data = {}
# outliers_data = {}

# for column in columns:
#     column_data = data[column].dropna().tolist()
#     q1 = np.percentile(column_data, 25) 
#     q3 = np.percentile(column_data, 75) 
#     iqr = q3 - q1 
#     lower_bound = q1 - 1.5 * iqr 
#     upper_bound = q3 + 1.5 * iqr  
    
#     outliers = [x for x in column_data if x < lower_bound or x > upper_bound]
#     non_outliers = [x for x in column_data if lower_bound <= x <= upper_bound]
    
#     outliers_data[column] = outliers
#     clean_data[column] = non_outliers
# print(outliers_data)
# chart = lc.BoxPlot(
#     data=clean_data,
#     theme=lc.Themes.Dark,  
#     title='Sea Metrics Distribution (Excluding Outliers)',
#     xlabel='Metric',
#     ylabel='Values'
# )
# chart.set_series_background_color(lc.Color(0, 255, 255))
# x_coordinates = {column: idx for idx, column in enumerate(columns)}

# cat_i =0
# for column, y_values in outliers_data.items():
#     series = chart.add_point_series(
#         sizes=True,
#         rotations=True,
#         lookup_values=True
#     )
#     series.append_samples(
#         x_values=[0.5 + cat_i] * len(y_values),  
#         y_values=y_values,
#         sizes=[10] * len(y_values) 
#     )
#     series.set_individual_point_color_enabled()
#     series.set_point_color(lc.Color('red'))  
#     series.set_point_shape("triangle") 
#     cat_i += 2

# chart.set_cursor_mode("show-nearest")

# chart.open()






import pandas as pd
import numpy as np
import lightningchart as lc

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

columns = ['temperature', 'windspeed', 'sigheight', 'humidity', 'feelslike', 'swellheight']

dataset=[]
x_values_outlier=[]
y_values_outlier=[]

chart = lc.ChartXY(
    theme=lc.Themes.Dark,
    title='Box Series'
)
chart.set_series_background_color(lc.Color(0, 255, 255))
for i,column in enumerate(columns):
    column_data = data[column].dropna().tolist()
    start=(i * 2) + 1 
    end=start + 1
    lowerQuartile = float(np.percentile(column_data, 25)) 
    upperQuartile = float(np.percentile(column_data, 75)) 
    median = float(np.median(column_data))
    lowerExtreme = float(np.min(column_data))
    upperExtreme = float(np.max(column_data))  
    # print(f'{column}: {lowerQuartile, upperQuartile, median, lowerExtreme, upperExtreme}')       
    dic={'start':start,'end':end,'lowerQuartile':lowerQuartile,'upperQuartile':upperQuartile,'median':median,'lowerExtreme':lowerExtreme,'upperExtreme':upperExtreme}
    dataset.append(dic)
    iqr = upperQuartile - lowerQuartile
    lower_bound = lowerQuartile - 1.5 * iqr
    upper_bound = upperQuartile + 1.5 * iqr
    outliers = [y for y in column_data if y < lower_bound or y > upper_bound]

    for outlier in outliers:
        x_values_outlier.append(start + 0.5) 
        y_values_outlier.append(outlier)

# print(dataset)
# print(y_values_outlier)
series = chart.add_box_series()
series.add_multiple(dataset)


series = chart.add_point_series(
        sizes=True,
        rotations=True,
        lookup_values=True
    )
series.set_point_color(lc.Color('red'))
series.set_point_shape("triangle") 
series.append_samples(
    x_values=x_values_outlier,  
    y_values=y_values_outlier,
    sizes=[10] * len(y_values_outlier)
)

chart.open()



