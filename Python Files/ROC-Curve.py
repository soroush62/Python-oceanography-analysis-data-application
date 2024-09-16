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

# Set up your LightningChart license key
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load your dataset
file_path = 'Dataset/hour_forecast.csv'  
data = pd.read_csv(file_path)

# Define features and target variable
features = ['idhourforecast', 'iddayforecast', 'temperature', 'windspeed', 'winddirdegree', 'preciptation', 
            'humidity', 'pressure', 'cloundover', 'heatIndex', 'dewpoint', 'windchill', 'windgust', 
            'feelslike', 'swellheight', 'swelldir', 'period', 'watertemp']
target = 'sigheight'

# Transform the target variable into binary classification for ROC
y = pd.cut(data[target], bins=[0, 0.5, 1.5], labels=['Low', 'High']).apply(lambda x: 1 if x == 'High' else 0)

X = data[features]

# Preprocessing (Standardizing numerical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

# Define the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Dashboard
dashboard = lc.Dashboard(rows=2, columns=3, theme=lc.Themes.Dark)

# Function to add ROC curve to the dashboard
def add_roc_curve_to_dashboard(dashboard, model_name, model, column_index, row_index):    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    thresholds = np.nan_to_num(thresholds, nan=0.0)
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    
    # Create chart
    chart = dashboard.ChartXY(column_index=column_index, row_index=row_index, column_span=1, row_span=1)
    chart.set_title(f'{model_name} ROC Curve (AUC = {roc_auc:.2f})')
    
    # ROC curve
    roc_series = chart.add_line_series()
    roc_series.add(fpr.tolist(), tpr.tolist()).set_name('ROC Curve')
    
    # Diagonal line (random guess)
    diagonal_series = chart.add_line_series()
    diagonal_series.add([0, 1], [0, 1])
    diagonal_series.set_name('Chance')
    diagonal_series.set_dashed(pattern='Dashed')
    
    # Add threshold points
    point_series = chart.add_point_series().set_name('Threshold Points')
    for j in range(len(thresholds)):
        color = lc.Color(
            int(255 * (1 - normalized_thresholds[j])),  # Red
            int(255 * normalized_thresholds[j]),        # Green
            0                                           # Blue
        )
        point_series.add(fpr[j], tpr[j]).set_point_color(color)
    
    # Set axis titles
    chart.get_default_x_axis().set_title('False Positive Rate')
    chart.get_default_y_axis().set_title('True Positive Rate')
    
    # Add legend
    legend = chart.add_legend(horizontal=False)
    legend.add(roc_series)
    legend.add(point_series)

# Add ROC curve to the dashboard for each model
for i, (model_name, model) in enumerate(models.items()):
    add_roc_curve_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

# Ensemble Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

# Add ROC curve for ensemble classifier
add_roc_curve_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

# Open the dashboard
dashboard.open()
