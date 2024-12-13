import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt


file_path = './bank--additional-full.csv'
data = pd.read_csv(file_path, delimiter=';')

# 1. Ознайомлення з даними та визначення цільової змінної
target_column = 'y' 
data[target_column] = data[target_column].map({'yes': 1, 'no': 0}) 

# 2. Попередня обробка даних
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

X = data_encoded.drop(columns=[target_column])
y = data_encoded[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Побудова ансамблевої моделі AdaBoost

base_model = DecisionTreeClassifier(max_depth=1)

# Вибірка гіперпараметрів
param_range = np.arange(10, 500, 10)
train_scores, test_scores = validation_curve(
    AdaBoostClassifier(estimator=base_model, learning_rate=0.5, random_state=42),
    X_train,
    y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Побудова валідаційної кривої
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(param_range, train_mean, label="Training Score")
plt.plot(param_range, test_mean, label="Validation Score")
plt.title("Validation Curve with AdaBoost")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()


# 4. Побудова ансамблевої моделі GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

# Базова модель - неглибоке дерево рішень
learning_rate = 0.1  # Фіксоване значення
param_range = np.arange(10, 500, 10)

# Валідаційна крива для GradientBoosting
train_scores, test_scores = validation_curve(
    GradientBoostingClassifier(max_depth=1, learning_rate=learning_rate, random_state=42),
    X_train,
    y_train,
    param_name="n_estimators",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Побудова валідаційної кривої
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(param_range, train_mean, label="Training Score")
plt.plot(param_range, test_mean, label="Validation Score")
plt.title("Validation Curve with GradientBoosting")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()


# 5. Побудова моделі XGBoost та LightGBM з оптимізацією

# Побудова моделі XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Підбір гіперпараметрів для XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [1, 3, 5]
}

xgb_grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=xgb_param_grid, scoring='accuracy', cv=5, n_jobs=-1
)
xgb_grid_search.fit(X_train, y_train)

# Найкращі параметри для XGBoost
xgb_best_model = xgb_grid_search.best_estimator_
print("Найкращі параметри для XGBoost:", xgb_grid_search.best_params_)
print("Точність на валідаційних даних (XGBoost):", xgb_grid_search.best_score_)

# Побудова моделі LightGBM з оптимізованими параметрами
lgbm_model = LGBMClassifier(
    force_row_wise=True, 
    min_data_in_leaf=50,  
    min_split_gain=0.2,   
    num_leaves=15,       
    random_state=42
)


# Підбір гіперпараметрів для LightGBM
lgbm_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [1, 3, 5]
}

lgbm_grid_search = GridSearchCV(
    estimator=lgbm_model, param_grid=lgbm_param_grid, scoring='accuracy', cv=5, n_jobs=-1
)
lgbm_grid_search.fit(X_train, y_train)

# Найкращі параметри для LightGBM
lgbm_best_model = lgbm_grid_search.best_estimator_
print("Найкращі параметри для LightGBM:", lgbm_grid_search.best_params_)
print("Точність на валідаційних даних (LightGBM):", lgbm_grid_search.best_score_)

# Аналіз топ-10 ознак для XGBoost
importances_xgb = pd.Series(xgb_best_model.feature_importances_, index=X_train.columns)
top_10_xgb = importances_xgb.nlargest(10)

# Побудова графіка для XGBoost
plt.figure(figsize=(10, 6))
top_10_xgb.plot(kind='barh', color='skyblue')
plt.title("Top 10 Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()

# Аналіз топ-10 ознак для LightGBM
importances_lgbm = pd.Series(lgbm_best_model.feature_importances_, index=X_train.columns)
top_10_lgbm = importances_lgbm.nlargest(10)

# Побудова графіка для LightGBM
plt.figure(figsize=(10, 6))
top_10_lgbm.plot(kind='barh', color='lightgreen')
plt.title("Top 10 Feature Importance (LightGBM)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()
