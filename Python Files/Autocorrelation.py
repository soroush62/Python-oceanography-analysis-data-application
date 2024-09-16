import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Set up your LightningChart license key
with open('D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt', 'r') as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load your dataset
file_path = 'Dataset/hour_forecast.csv'
data = pd.read_csv(file_path)

# Select relevant features and target variable
selected_features = ['temperature', 'windspeed', 'humidity', 'feelslike', 'swellheight', 'watertemp']  # Modify based on your dataset
target_variable = 'sigheight'  # Replace this with your target variable

X = data[selected_features]
y = data[target_variable]

# Preprocessing step: Standardize numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features)
    ])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
nb_model = GaussianNB()

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', nb_model)
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict probabilities for ROC and PR Curves
y_scores = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

# Create Dashboard
dashboard = lc.Dashboard(rows=1, columns=2, theme=lc.Themes.Dark)

# ROC Curve Chart
roc_chart = dashboard.ChartXY(row_index=0, column_index=0)
roc_chart.set_title(f'Naive Bayes ROC Curve (AUC = {roc_auc:.2f})')

roc_series = roc_chart.add_line_series()
roc_series.add(fpr.tolist(), tpr.tolist()).set_name('ROC Curve')

roc_chart.get_default_x_axis().set_title('False Positive Rate')
roc_chart.get_default_y_axis().set_title('True Positive Rate')

# Precision-Recall Curve Chart
pr_chart = dashboard.ChartXY(row_index=0, column_index=1)
pr_chart.set_title(f'Naive Bayes Precision-Recall Curve (AUC = {pr_auc:.2f})')

pr_series = pr_chart.add_line_series()
pr_series.add(recall.tolist(), precision.tolist()).set_name('Precision-Recall Curve')

pr_chart.get_default_x_axis().set_title('Recall')
pr_chart.get_default_y_axis().set_title('Precision')

# Open the dashboard
dashboard.open()
