import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, mean_squared_error


df = pd.read_csv('D:/Ml fall-2024/ml_Fall2024/LW2/bank--additional-full.csv', sep=';')

print(f'Info about the dataframe:\n{df.info()}')
print(f'First five rows of the dataframe:\n{df.head()}')
print(f'Shape of data:\n{df.shape}')

df.replace({
    'unknown': np.nan,
    'nonexistent': np.nan,
    999: np.nan
}, inplace=True)


print(f'Number of missing values in columns:\n{df.isnull().sum()}')
print(f'Total number of missing values:\n{df.isnull().sum().sum()}')


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f'Number of duplicated rows:\n{df.duplicated().value_counts()}')
print(f'Info about the numerical columns:\n{df.describe()}')
print(f'Info about categorical columns:\n{df.describe(include=["object"])}')


scaler = StandardScaler()
df[['duration', 'previous']] = scaler.fit_transform(df[['duration', 'previous']])

label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])
    print(f'{col} encoding classes: {label_enc.classes_}')

print(df.head())

X = df.drop(columns=['y'])
y = df['y']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


def train_and_evaluate_model(model, X_train, y_train, X_dev, y_dev):
    model.fit(X_train, y_train)
    y_dev_pred = model.predict(X_dev)
    metrics = {}
    if isinstance(model, (LogisticRegression, RidgeClassifier)):
        metrics['accuracy'] = accuracy_score(y_dev, y_dev_pred)
        metrics['precision'] = precision_score(y_dev, y_dev_pred)
        metrics['recall'] = recall_score(y_dev, y_dev_pred)
        metrics['f1'] = f1_score(y_dev, y_dev_pred)
        metrics['roc_auc'] = roc_auc_score(y_dev, y_dev_pred)
        metrics['conf_matrix'] = confusion_matrix(y_dev, y_dev_pred)
    else:
        metrics['Mean Squared Error'] = mean_squared_error(y_dev, y_dev_pred)
    return metrics

models = {
    'Logistic Regression (Ridge)': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000),
    'Ridge Classifier': RidgeClassifier(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}
param_grid = {
    'Logistic Regression (Ridge)': {'C': np.logspace(-3, 3, 10)},
    'Ridge Classifier': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'ElasticNet': {'alpha': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}
}


best_models = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy' if isinstance(model, (LogisticRegression, RidgeClassifier)) else 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f'Best parameters for {name}: {grid_search.best_params_}')

for name, model in best_models.items():
    y_test_pred = model.predict(X_test)
    metrics = {}
    if isinstance(model, (LogisticRegression, RidgeClassifier)):
        metrics['accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['precision'] = precision_score(y_test, y_test_pred)
        metrics['recall'] = recall_score(y_test, y_test_pred)
        metrics['f1'] = f1_score(y_test, y_test_pred)
        metrics['roc_auc'] = roc_auc_score(y_test, y_test_pred)
        metrics['conf_matrix'] = confusion_matrix(y_test, y_test_pred)
    else:
        metrics['Mean Squared Error '] = mean_squared_error(y_test, y_test_pred)
    
    print(f'Metrics for {name} on test set:')
    for metric, value in metrics.items():
        if metric != 'conf_matrix':
            print(f'{metric.capitalize()}: {value:.2f}')
        else:
            print(f'Confusion Matrix:\n{value}')



for name, model in best_models.items():
    metrics = train_and_evaluate_model(model, X_train, y_train, X_dev, y_dev)
    print(f'Metrics for {name} on dev set:')
    for metric, value in metrics.items():
        if metric != 'conf_matrix':
            print(f'{metric.capitalize()}: {value:.2f}')
        else:
            print(f'Confusion Matrix:\n{value}')


def plot_validation_curves(models, param_grid, X_train, y_train):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    idx = 0
    for name, model in models.items():
        for param_name, param_range in param_grid[name].items():
            train_scores, valid_scores = validation_curve(
                model, X_train, y_train,
                param_name=param_name,
                param_range=param_range,
                cv=5,
                scoring='accuracy' if isinstance(model, (LogisticRegression, RidgeClassifier)) else 'neg_mean_squared_error'
            )
            
            row, col = divmod(idx, 2)
            axes[row, col].plot(param_range, np.mean(train_scores, axis=1), label='Train Score', color='blue')
            axes[row, col].plot(param_range, np.mean(valid_scores, axis=1), label='Validation Score', color='orange')
            axes[row, col].set_xscale('log')
            axes[row, col].set_xlabel(param_name)
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_title(f'{name} ({param_name})')
            axes[row, col].legend(loc='best')
            axes[row, col].grid(True)
            
            idx += 1

    for i in range(idx, 6):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

plot_validation_curves(models, param_grid, X_train, y_train)




