import pandas as pd
import numpy as np
import lightningchart as lc

lc.set_license(open('../license-key').read())

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
    dic={'start':start,'end':end,'lowerQuartile':lowerQuartile,'upperQuartile':upperQuartile,'median':median,'lowerExtreme':lowerExtreme,'upperExtreme':upperExtreme}
    dataset.append(dic)
    iqr = upperQuartile - lowerQuartile
    lower_bound = lowerQuartile - 1.5 * iqr
    upper_bound = upperQuartile + 1.5 * iqr
    outliers = [y for y in column_data if y < lower_bound or y > upper_bound]

    for outlier in outliers:
        x_values_outlier.append(start + 0.5) 
        y_values_outlier.append(outlier)

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

