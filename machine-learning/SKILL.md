---
name: machine-learning
description: Comprehensive machine learning guidance covering ML algorithms, deep learning, model training, evaluation, deployment, MLOps, and production best practices. Use when building ML models, training neural networks, implementing ML pipelines, or deploying ML systems to production.
---

# Machine Learning

Expert guidance for building, training, evaluating, and deploying machine learning models following industry best practices and modern MLOps principles.

## Machine Learning Workflow

```
Data Collection → Data Preprocessing → Feature Engineering → Model Selection → 
Training → Evaluation → Hyperparameter Tuning → Deployment → Monitoring
```

## Data Preprocessing

### Data Loading and Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic exploration
print(df.info())
print(df.describe())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Visualize distributions
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.show()
```

### Handling Missing Data
```python
from sklearn.impute import SimpleImputer, KNNImputer

# Remove rows with missing values
df_cleaned = df.dropna()

# Fill with mean/median/mode
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# KNN Imputation for better results
knn_imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])

# Forward/backward fill
df.fillna(method='ffill', inplace=True)
```

### Handling Outliers
```python
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(df[numerical_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]

# IQR method
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | 
                       (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling (0 to 1)
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

# Robust scaling (handles outliers)
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)
```

### Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding (ordinal)
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# One-Hot Encoding (nominal)
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Target Encoding
target_means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_means)
```

## Feature Engineering

### Creating New Features
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                          labels=['child', 'young', 'middle', 'senior'])

# Interaction features
df['feature_interaction'] = df['feature1'] * df['feature2']
```

### Feature Selection
```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Feature importance from tree-based models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# L1-based feature selection
from sklearn.linear_model import Lasso
selector = SelectFromModel(Lasso(alpha=0.01))
X_selected = selector.fit_transform(X, y)
```

## Classical Machine Learning Algorithms

### Linear Models
```python
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
```

### Tree-Based Models
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

# LightGBM (faster for large datasets)
lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
lgbm.fit(X_train, y_train)

# CatBoost (handles categorical features automatically)
catboost = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    verbose=False
)
catboost.fit(X_train, y_train)
```

### Support Vector Machines
```python
from sklearn.svm import SVC, SVR

# Classification
svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)

# Regression
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr.fit(X_train, y_train)
```

### K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
```

### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Multinomial Naive Bayes (for text classification)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
```

## Model Evaluation

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Basic metrics
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ROC-AUC
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, cross_validate

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)
for metric in scoring:
    print(f"{metric}: {scores[f'test_{metric}'].mean():.4f}")
```

## Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

### Bayesian Optimization
```python
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10)
}

bayes_search = BayesSearchCV(
    GradientBoostingClassifier(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
print(f"Best parameters: {bayes_search.best_params_}")
```

## Deep Learning with PyTorch

### Neural Network Architecture
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_sizes=[128, 64, 32],
    output_size=num_classes
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
```

### Training Loop
```python
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, device='cuda'):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model
```

### Convolutional Neural Networks (CNN)
```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

### Recurrent Neural Networks (RNN/LSTM)
```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = self.fc(out[:, -1, :])
        return out
```

### Transfer Learning
```python
import torchvision.models as models
import torch.nn as nn

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Only train the final layers
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

## MLOps and Production

### Model Serialization
```python
import joblib
import pickle

# Scikit-learn models
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')

# PyTorch models
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Model Versioning with MLflow
```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

### Model Deployment with FastAPI
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model at startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: float
    probability: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Preprocess
        features = np.array(input_data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()
        
        return PredictionOutput(
            prediction=float(prediction),
            probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Docker Containerization
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Model Monitoring
```python
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')

@prediction_latency.time()
def make_prediction(features):
    prediction_counter.inc()
    
    # Make prediction
    result = model.predict(features)
    
    # Log prediction
    logger.info(f"Prediction made at {datetime.now()}: {result}")
    
    return result

# Monitor data drift
from scipy.stats import ks_2samp

def check_data_drift(reference_data, current_data, threshold=0.05):
    """Check for data drift using Kolmogorov-Smirnov test."""
    drift_detected = {}
    
    for col in reference_data.columns:
        statistic, p_value = ks_2samp(reference_data[col], current_data[col])
        drift_detected[col] = p_value < threshold
        
        if drift_detected[col]:
            logger.warning(f"Data drift detected in column {col}")
    
    return drift_detected
```

## Best Practices

### Data Management
- [ ] Version control your data with DVC or similar tools
- [ ] Document data sources and transformations
- [ ] Validate data quality regularly
- [ ] Handle imbalanced datasets (SMOTE, class weights)
- [ ] Split data chronologically for time-series
- [ ] Use stratified splits for classification

### Model Development
- [ ] Start with simple models before complex ones
- [ ] Use cross-validation to prevent overfitting
- [ ] Track experiments with MLflow or Weights & Biases
- [ ] Save preprocessing pipelines with models
- [ ] Document model assumptions and limitations
- [ ] Implement proper error handling

### Training
- [ ] Set random seeds for reproducibility
- [ ] Use early stopping to prevent overfitting
- [ ] Monitor training and validation metrics
- [ ] Save checkpoints during training
- [ ] Use learning rate scheduling
- [ ] Implement gradient clipping for stability

### Evaluation
- [ ] Use multiple metrics (don't rely on accuracy alone)
- [ ] Test on holdout set separate from validation
- [ ] Analyze error cases and failure modes
- [ ] Check for bias in predictions
- [ ] Validate on production-like data
- [ ] Perform A/B testing when possible

### Deployment
- [ ] Version models in production
- [ ] Implement monitoring and alerting
- [ ] Log predictions and features
- [ ] Set up CI/CD pipeline
- [ ] Use containerization (Docker)
- [ ] Implement graceful degradation
- [ ] Plan for model retraining

### Security
- [ ] Sanitize user inputs
- [ ] Implement rate limiting
- [ ] Use authentication for API endpoints
- [ ] Encrypt sensitive data
- [ ] Monitor for adversarial attacks
- [ ] Follow data privacy regulations (GDPR, etc.)

## Common Pitfalls

### Data Leakage
```python
# Wrong - fitting scaler on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)

# Correct - fit only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Overfitting
```python
# Signs of overfitting
- High training accuracy, low test accuracy
- Large gap between training and validation loss
- Model performs well on training data but poorly on new data

# Solutions
- Use regularization (L1/L2)
- Reduce model complexity
- Increase training data
- Use dropout (for neural networks)
- Apply data augmentation
- Early stopping
```

### Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class weights
class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train),
                                      y=y_train)
model = RandomForestClassifier(class_weight='balanced')
```

## Tools and Libraries

### Essential Libraries
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Classical ML algorithms
- **PyTorch/TensorFlow**: Deep learning
- **XGBoost/LightGBM/CatBoost**: Gradient boosting
- **Matplotlib/Seaborn**: Visualization
- **MLflow**: Experiment tracking
- **FastAPI**: Model deployment
- **Docker**: Containerization

### Advanced Tools
- **Weights & Biases**: Experiment tracking
- **Optuna/Hyperopt**: Hyperparameter optimization
- **SHAP/LIME**: Model interpretability
- **DVC**: Data version control
- **Kubeflow**: ML workflows on Kubernetes
- **Ray**: Distributed computing
- **Triton**: Model serving

## Resources

### Online Courses
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai Deep Learning Course
- Deep Learning Specialization (Coursera)

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Hundred-Page Machine Learning Book" by Andriy Burkov

### Competitions
- Kaggle
- DrivenData
- AIcrowd

### Papers and Research
- arXiv.org (ML section)
- Papers With Code
- Google Scholar

## Performance Optimization

### Memory Management
```python
# Use generators for large datasets
def data_generator(batch_size=32):
    while True:
        batch = load_next_batch(batch_size)
        yield batch

# Use efficient data types
df['category'] = df['category'].astype('category')
df['integer_col'] = df['integer_col'].astype('int32')

# Delete unused variables
del large_dataframe
import gc
gc.collect()
```

### Training Optimization
```python
# Mixed precision training (PyTorch)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient accumulation
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Model Interpretability

### Feature Importance
```python
import shap

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)

# Force plot for single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

### LIME
```python
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['class_0', 'class_1'],
    mode='classification'
)

explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba
)

explanation.show_in_notebook()
```
