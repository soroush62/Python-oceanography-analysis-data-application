import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc

# Set up your LightningChart license key
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable2.txt', 'r') as f:
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

# Transform the target variable into binary classification for Precision-Recall
y = pd.cut(data[target], bins=[0, 0.5, 1.5], labels=['Low', 'High']).apply(lambda x: 1 if x == 'High' else 0)

X = data[features]

# Preprocessing (Standardizing numerical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Dashboard
dashboard = lc.Dashboard(columns=3, rows=2, theme=lc.Themes.Dark)

# Function to add Precision-Recall curve to the dashboard
def add_pr_curve_to_dashboard(dashboard, model_name, model, column_index, row_index):    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    # Compute Precision-Recall curve and AUC
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    thresholds = np.nan_to_num(thresholds, nan=0.0)
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    
    # Create chart
    chart = dashboard.ChartXY(column_index=column_index, row_index=row_index, column_span=1, row_span=1)
    chart.set_title(f'{model_name} Precision-Recall Curve (AUC = {pr_auc:.2f})')
    
    # Precision-Recall curve
    pr_series = chart.add_line_series()
    pr_series.add(recall.tolist(), precision.tolist()).set_name('Precision-Recall Curve')
    
    # Add threshold points
    point_series = chart.add_point_series().set_name('Threshold Points')
    for j in range(len(thresholds)):
        color = lc.Color(
            int(255 * (1 - normalized_thresholds[j])),  # Red
            int(255 * normalized_thresholds[j]),        # Green
            0                                           # Blue
        )
        point_series.add(recall[j], precision[j]).set_point_color(color)
    
    # Set axis titles
    chart.get_default_x_axis().set_title('Recall')
    chart.get_default_y_axis().set_title('Precision')
    
    # Add legend
    legend = chart.add_legend(horizontal=False)
    legend.add(pr_series)
    legend.add(point_series)

# Add Precision-Recall curve to the dashboard for each model
for i, (model_name, model) in enumerate(models.items()):
    add_pr_curve_to_dashboard(dashboard, model_name, model, column_index=i % 3, row_index=i // 3)

# Ensemble Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

# Add Precision-Recall curve for ensemble classifier
add_pr_curve_to_dashboard(dashboard, 'Ensemble Methods', voting_clf, column_index=2, row_index=1)

# Open the dashboard
dashboard.open()
