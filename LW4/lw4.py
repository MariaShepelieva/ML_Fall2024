import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, validation_curve
from sklearn.metrics import  accuracy_score, accuracy_score, classification_report
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pydotplus

df = pd.read_csv('./bank--additional-full.csv', sep=';')

df = df.drop(columns=['duration'])

label_encoder = LabelEncoder()

for col in ['default', 'housing', 'loan', 'y']:
    df[col] = label_encoder.fit_transform(df[col])

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier(
    min_samples_split = 5, 
    max_features= 10, 
    random_state=42)

classifier.fit(X_train, y_train)
y_pred_class = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Точність класифікації Дерева рішень (accuracy): {accuracy:.4f}")


kf = KFold( n_splits = 5, shuffle = True, random_state = 42 )



param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt', 'log2']  
}



clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=kf, scoring='accuracy')
clf.fit(X_train, y_train)
print(f"Найкращі параметри: {clf.best_params_}")
print(f"Точність GridSearchCV : {clf.best_score_:.4f}")


def plot_validation_curve(param_range, train_scores, test_scores, param_name, file_name):

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Навчальна вибірка", marker='o', color='#FA8072')
    plt.plot(param_range, test_mean, label="Тестова вибірка", marker='o', color='#2E8B57')
    plt.title(f"Валідаційна крива для DecisionTreeClassifier ({param_name})")
    plt.xlabel(param_name)
    plt.ylabel("Точність (Accuracy)")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(file_name)
    plt.close()


clf = DecisionTreeClassifier(random_state=42, min_samples_split=5)
X_train, y_train

param_ranges = {

    "max_depth": [3, 5, 10, 15, 20, None],
    "min_samples_split": [1, 2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 4, 6, 8],
}

file_names = {

    "max_depth": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_decision_tree_max_depth.png',
    "min_samples_split": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_decision_tree_min_samples_split.png',
    "min_samples_leaf": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_decision_tree_min_samples_leaf.png',
}


for param_name, param_range in param_ranges.items():
    train_scores, test_scores = validation_curve(
        clf, X_train, y_train,
        param_name=param_name, 
        param_range=param_range,
        cv=kf, 
        scoring="accuracy"
    )

    plot_validation_curve(param_range, train_scores, test_scores, param_name, file_names[param_name])

param_range_1 = ['None', 'sqrt', 'log2']

train_scores_clf, test_scores_clf= validation_curve(

    DecisionTreeClassifier(random_state=42, min_samples_split = 5),
    X_train, y_train,
    param_name='max_features',
    param_range=[None, 'sqrt', 'log2'],
    cv=kf,
    scoring="accuracy"
)

train_mean = np.mean(train_scores_clf,  axis=1)
test_mean = np.mean(test_scores_clf, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range_1, train_mean, label="Навчальна вибірка", marker='o', color='#FA8072')
plt.plot(param_range_1, test_mean, label="Тестова вибірка", marker='o', color='#2E8B57')
plt.title("Валідаційна крива для DecisionTreeClassifier")
plt.xlabel("max_features")
plt.ylabel("Точність (Accuracy)")
plt.legend(loc="best")
plt.grid()
plt.savefig('/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_decision_tree_max_features.png')
plt.close()



X_unscaled = df.drop('y', axis=1)


classifier = DecisionTreeClassifier(min_samples_split=5, max_features=10, random_state=42)
classifier = DecisionTreeClassifier(

    max_depth=4,            
    min_samples_split=10,    
    min_samples_leaf=5,      
    random_state=42
)

classifier.fit(X_train, y_train)

dot_data = export_graphviz(

    classifier,
    out_file=None,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = pydotplus.graph_from_dot_data(dot_data)
output_path = '/Machine Learning Course/ML_Fall2024/LW4/Results/compact_decision_tree.png'
graph.write_png(output_path)
print(f"Графічне представлення компактного дерева збережено як '{output_path}'.")

def plot_feature_importance(model, X):
    feature_importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance')
    plt.savefig('/Machine Learning Course/ML_Fall2024/LW4/Results/feature_importances.png')
    plt.close()

plot_feature_importance(classifier, X)


def plot_validation_curve(param_range, train_scores, test_scores, param_name, file_name):
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Навчальна вибірка", marker='o', color='#FA8072')
    plt.plot(param_range, test_mean, label="Тестова вибірка", marker='o', color='#2E8B57')
    plt.title(f"Валідаційна крива для RandomForestClassifier ({param_name})")
    plt.xlabel(param_name)
    plt.ylabel("Точність (Accuracy)")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(file_name)
    plt.close()

param_ranges = {

    "max_depth": [3, 5, 10, 15, 20, None],
    "min_samples_split": [2, 3, 4, 5], 
    "n_estimators": [10, 20, 30, 65, 80],
}

file_names = {

    "max_depth": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_random_forest_max_depth.png',
    "min_samples_split": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_random_forest_min_samples_split.png',
    "n_estimators": '/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_random_forest_n_estimators.png',
}


clf = RandomForestClassifier(random_state=42, min_samples_split=5)
clf.fit(X_train, y_train)
y_pred_class = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Точність класифікації Випадкового лісу (accuracy): {accuracy:.4f}")

for param_name, param_range in param_ranges.items():
    train_scores, test_scores = validation_curve(

        clf, X_train, y_train,
        param_name=param_name, param_range=param_range,
        cv=kf, scoring="accuracy"

    )
    plot_validation_curve(param_range, train_scores, test_scores, param_name, file_names[param_name])


param_range_1 = ['None', 'sqrt', 'log2']
train_scores_clf, test_scores_clf = validation_curve(

    RandomForestClassifier(
        random_state=42, 
        min_samples_split=5
        ),
    X_train,
    y_train,
    param_name='max_features',
    param_range=[None, 'sqrt', 'log2'],
    cv=kf,
    scoring="accuracy"
)


train_mean = np.mean(train_scores_clf, axis=1)
test_mean = np.mean(test_scores_clf, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(param_range_1, train_mean, label="Навчальна вибірка", marker='o', color='#FA8072')
plt.plot(param_range_1, test_mean, label="Тестова вибірка", marker='o', color='#2E8B57')
plt.title("Валідаційна крива для RandomForestClassifier (max_features)")
plt.xlabel("max_features")
plt.ylabel("Точність (Accuracy)")
plt.legend(loc="best")
plt.grid()
plt.savefig('/Machine Learning Course/ML_Fall2024/LW4/Results/validation_curve_random_forest_max_features.png')
plt.close()


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,            
    random_state=42          
)
rf_model.fit(X_train, y_train)

feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="pink")
plt.title("Топ-10 найкорисніших ознак (RandomForestClassifier)", fontsize=16)
plt.xlabel("Важливість ознаки", fontsize=14)
plt.ylabel("Ознаки", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/Machine Learning Course/ML_Fall2024/LW4/Results/top_10_feature_importances.png')
plt.close()

knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)


dt_model = DecisionTreeClassifier(
    max_depth=10, 
    min_samples_split=5, 
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)


print("=== Порівняння точності моделей ===")
print(f"K-Nearest Neighbors: {knn_accuracy:.4f}")
print(f"Decision Tree:        {dt_accuracy:.4f}")
print(f"Random Forest:        {rf_accuracy:.4f}")

print("\n=== Звіт для K-Nearest Neighbors ===")
print(classification_report(y_test, y_pred_knn))
print("\n=== Звіт для Decision Tree ===")
print(classification_report(y_test, y_pred_dt))
print("\n=== Звіт для Random Forest ===")
print(classification_report(y_test, y_pred_rf))