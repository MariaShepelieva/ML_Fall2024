import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler



df = pd.read_csv('ML_Fall2024/LW5/bank--additional-full.csv', sep = ";", na_values='unknown')

print("Перші рядки даних:")
print(df.head())
print("\nІнформація про датафрейм:")
print(df.info())
print("\nКількість пропущених значень у кожному стовпці:")
print(df.isna().sum())

na_columns = ['job','marital','education','default','housing','loan']
for col in na_columns:
    df[col] = df[col].fillna(df[col].mode().values[0])

print("\nКількість пропущених значень після обробки:")
print(df.isna().sum())

# Статистичний опис даних
print("\nСтатистичний опис числових стовпців:")
print(df.describe())



q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3-q1
df = df[( df['age'] > q1 - 1.5 * iqr) & (df['age'] < q3+1.5 * iqr)]
print(df)



label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])
for column in ['job', 'marital', 'education','default','housing','loan','contact','month','day_of_week','poutcome']:
    df[column] = label_encoder.fit_transform(df[column])

print(df.describe(include='all'))

X = df.drop('y', axis=1)  
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 1. XGBoost модель
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 2. LightGBM модель
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42
)

# Гіперпараметри для аналізу
n_estimators_range = [50, 100, 200, 300, 400]
learning_rate_range = [0.01, 0.05, 0.1, 0.2]

# Валідаційні криві для n_estimators
def plot_validation_curve(model, X_train, y_train, param_name, param_range, scoring, title):
    train_scores, test_scores = validation_curve(
        model, X_train, y_train,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=5
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Training score", color="blue", marker="o")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="orange", marker="o")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Валідаційна крива для XGBoost (n_estimators)
plot_validation_curve(xgb_model, X_train, y_train, 'n_estimators', n_estimators_range, 'f1', 'Validation Curve for XGBoost (n_estimators)')

# Валідаційна крива для LightGBM (learning_rate)
plot_validation_curve(lgb_model, X_train, y_train, 'learning_rate', learning_rate_range, 'f1', 'Validation Curve for LightGBM (learning_rate)')

# Навчання моделей з оптимальними гіперпараметрами
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Передбачення та метрики
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

print("\nLightGBM Classification Report:")
print(classification_report(y_test, lgb_pred))

# Аналіз важливості ознак
def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Побудова графіків важливості ознак
plot_feature_importance(xgb_model, X.columns, "XGBoost Feature Importance")
plot_feature_importance(lgb_model, X.columns, "LightGBM Feature Importance")
