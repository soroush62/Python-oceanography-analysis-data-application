import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

file_path = 'Dataset/hour_forecast.csv'  
data = pd.read_csv(file_path)

features = ['idhourforecast', 'iddayforecast', 'temperature', 'windspeed', 'winddirdegree', 'preciptation', 
            'humidity', 'pressure', 'cloundover', 'heatIndex', 'dewpoint', 'windchill', 'windgust', 
            'feelslike', 'swellheight', 'swelldir', 'period', 'watertemp']
target = 'sigheight'

data = data.dropna(subset=[target])
min_value = data[target].min()
max_value = data[target].max()
mid_value = (min_value + max_value) / 2

y = pd.cut(data[target], bins=[min_value, mid_value, max_value], labels=['Low', 'High'], include_lowest=True).apply(lambda x: 1 if x == 'High' else 0)

X = data[features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dashboard = lc.Dashboard(rows=2, columns=3, theme=lc.Themes.Dark)

def add_roc_curve_to_dashboard(dashboard, model_name, model, column_index, row_index):    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    thresholds = np.nan_to_num(thresholds, nan=0.0)
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    
    chart = dashboard.ChartXY(column_index=column_index, row_index=row_index, column_span=1, row_span=1)
    chart.set_title(f'{model_name} ROC Curve (AUC = {roc_auc:.2f})')
    
    roc_series = chart.add_line_series()
    roc_series.add(fpr.tolist(), tpr.tolist()).set_name('ROC Curve')
    
    diagonal_series = chart.add_line_series()
    diagonal_series.add([0, 1], [0, 1])
    diagonal_series.set_name('Chance')
    diagonal_series.set_dashed(pattern='Dashed')
    
    point_series = chart.add_point_series().set_name('Threshold Points')
    for j in range(len(thresholds)):
        color = lc.Color(
            int(255 * (1 - normalized_thresholds[j])),  # Red
            int(255 * normalized_thresholds[j]),        # Green
            0                                           # Blue
        )
        point_series.add(fpr[j], tpr[j]).set_point_color(color)
    
    chart.get_default_x_axis().set_title('False Positive Rate')
    chart.get_default_y_axis().set_title('True Positive Rate')
    
    legend = chart.add_legend(horizontal=False)
    legend.add(roc_series)
    legend.add(point_series)

for i, (model_name, model) in enumerate(models.items()):
    add_roc_curve_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

add_roc_curve_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

dashboard.open()
