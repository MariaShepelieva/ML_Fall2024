import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, mean_squared_error
import os

print(os.getcwd())
df = pd.read_csv('./bank--additional-full.csv', sep=';', na_values = 'unknown')

print(f'First five rows of the dataframe:\n{df.head()}')
print(f'Shape of data:\n{df.shape}')

label_encoder = LabelEncoder()

# Перетворимо бінарні категоріальні ознаки 'default', 'housing', 'loan', 'y' у числові
for col in ['default', 'housing', 'loan', 'y']:
    df[col] = label_encoder.fit_transform(df[col])

# Закодуємо інші категоріальні ознаки за допомогою one-hot кодування
df = pd.get_dummies(df, drop_first=True)

# Відокремлюємо ознаки (X) та цільову змінну (y)
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_model(model, X_train, y_train):
    model.fit(X_train, y_train)
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
        metrics['Mean Squared Error'] = mean_squared_error(y_test, y_test_pred)
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
    metrics = train_and_evaluate_model(model, X_train, y_train)
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


# Retrieve the best Logistic Regression model
best_model = best_models['Logistic Regression (Ridge)']

# Get feature names and coefficients
feature_names = X.columns
coefficients = best_model.coef_.flatten()  # Flatten the 2D array into a 1D array

# Create the DataFrame with feature names and coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Add absolute coefficient values for sorting
coef_df['AbsCoefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values(by='AbsCoefficient', ascending=False)

# Display the top coefficients
print("Top features sorted by importance:")
print(coef_df[['Feature', 'Coefficient']].head(20))

# Visualize the top 20 coefficients
top_20_coef = coef_df.head(20)
plt.figure(figsize=(10, 6))
plt.barh(top_20_coef['Feature'], top_20_coef['Coefficient'], color='b')
plt.xlabel('Coefficient Value')
plt.title('Top 20 Feature Importance Based on Coefficients')
plt.gca().invert_yaxis()  # Invert the y-axis for better readability
plt.grid(True)
plt.show()
