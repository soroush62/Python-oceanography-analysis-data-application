import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

lc.set_license(open('../license-key').read())

file_path = 'Dataset/hour_forecast.csv'  
data = pd.read_csv(file_path)

features = ['idhourforecast', 'iddayforecast', 'time', 'temperature', 'windspeed', 
            'winddirdegree', 'preciptation', 'humidity', 'pressure', 'cloundover', 
            'heatIndex', 'dewpoint', 'windchill', 'windgust', 'feelslike', 'swellheight', 
            'swelldir', 'period', 'watertemp']

bins = [0, 0.5, 1.0, 1.5]
labels = ['Low', 'Moderate', 'High']
data['sigheight_category'] = pd.cut(data['sigheight'], bins=bins, labels=labels, include_lowest=True)

category_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
data['sigheight_category'] = data['sigheight_category'].map(category_mapping)

X = data[features]
y = data['sigheight_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

dashboard = lc.Dashboard(rows=2, columns=3, theme=lc.Themes.Dark)

def add_feature_importance_to_dashboard(dashboard, model_name, model, column_index, row_index):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        importances = np.zeros(len(features))

    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    chart = dashboard.BarChart(
        column_index=column_index,
        row_index=row_index,
        column_span=1,
        row_span=1
    )
    chart.set_title(f'{model_name} Feature Importances')

    bar_data = [{'category': row['Feature'], 'value': row['Importance']} for _, row in importance_df.iterrows()]
    chart.set_data(bar_data)

for i, (model_name, model) in enumerate(models.items()):
    add_feature_importance_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

def add_ensemble_feature_importance_to_dashboard(dashboard, model_name, ensemble_model, column_index, row_index):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    ensemble_importances = np.zeros(len(features))
    for estimator in ensemble_model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            ensemble_importances += estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            ensemble_importances += np.abs(estimator.coef_[0])  
    
    ensemble_importances /= len(ensemble_model.estimators_)
    
    importance_df = pd.DataFrame({'Feature': features, 'Importance': ensemble_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].max()
    
    chart = dashboard.BarChart(
        column_index=column_index,
        row_index=row_index,
        column_span=1,
        row_span=1
    )
    chart.set_title(f'{model_name} Feature Importances')
    
    bar_data = [{'category': row['Feature'], 'value': row['Importance']} for _, row in importance_df.iterrows()]
    chart.set_data(bar_data)

add_ensemble_feature_importance_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

dashboard.open()