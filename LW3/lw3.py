import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.metrics import  accuracy_score, mean_squared_error, make_scorer, accuracy_score

df = pd.read_csv('./bank--additional-full.csv', sep=';')

print(f'Інформація про набір даних:\n{df.info()}')
print(f'Перші 5 стрічок набору даних:\n{df.head()}')
print(f'Форма подання даних:\n{df.shape}')

df.replace({
    'unknown': np.nan,
    'nonexistent': np.nan,
    999: np.nan
}, inplace=True)


print(f'Число пропущених значень в колонках:\n{df.isnull().sum()}')
print(f'Загальне число пропущених значень:\n{df.isnull().sum().sum()}')


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f'Кількість дублікатних стрічок:\n{df.duplicated().value_counts()}')
print(f'Інформація про числові дані(колонки):\n{df.describe()}')
print(f'Інформація про категоріальні дані (колонки):\n{df.describe(include=["object"])}')

sns.countplot(data=df, x='y', palette='pink')
plt.title("Розподіл значень цільової змінної (y)")
plt.xlabel("Підписка на строковий депозит")
plt.ylabel("Кількість клієнтів")
plt.show()


df['y'] = df['y'].map({'yes': 1, 'no': 0})


categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Розмір навчальної вибірки: X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'Розмір тестової вибірки: X_test: {X_test.shape}, y_test: {y_test.shape}')


y = df['y'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Точність класифікаційної моделі (accuracy): {accuracy:.4f}')


reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
print(f'Середньоквадратична похибка регресійної моделі (MSE): {mse:.4f}')


kf = KFold(n_splits=5, shuffle=True, random_state=42)


knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 51)}


grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X, y)


best_k = grid_search.best_params_['n_neighbors']
best_score = grid_search.best_score_
print(f'Найкраще значення k: {best_k}')
print(f'Оцінка якості при найкращому k: {best_score:.4f}')


k_values = range(1, 51)
scores = grid_search.cv_results_['mean_test_score']

plt.plot(k_values, scores, marker='o')
plt.xlabel('Кількість сусідів (k)')
plt.ylabel('Точність перехресної перевірки')
plt.title('Accuracy vs. Кількість сусідів (k) для kNN')
plt.grid()
plt.show()


p_values = np.linspace(1, 10, 20)
param_grid = {'n_neighbors': [best_k], 'p': p_values}
knn = KNeighborsClassifier()
grid_search_p = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
grid_search_p.fit(X, y)
best_p = grid_search_p.best_params_['p']
best_p_score = grid_search_p.best_score_

print(f'Найкраще значення p: {best_p:.4f}')
print(f'Оцінка якості при найкращому p: {best_p_score:.4f}')


p_values = grid_search_p.cv_results_['param_p']
scores = grid_search_p.cv_results_['mean_test_score']


plt.plot(p_values, scores, marker='o')
plt.xlabel('Значення p')
plt.ylabel('Точність на крос-валідації')
plt.title('Точність в залежності від p для KNN')
plt.grid()
plt.show()




knn_weighted = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
cv_scores = cross_val_score(knn_weighted, X, y, cv=5, scoring='accuracy')

print(f'Крос-валідаційна точність для KNN з вагами по відстані: {cv_scores}')
print(f'Середня точність на крос-валідації: {cv_scores.mean():.4f}')
print(f'Стандартне відхилення точності: {cv_scores.std():.4f}')



p_values = np.linspace(1, 10, 20)
mean_accuracies = []
for p in p_values:
    
    knn_weighted = KNeighborsClassifier(n_neighbors=best_k, p=p, weights='distance')
    
    
    cv_scores = cross_val_score(knn_weighted, X, y, cv=5, scoring='accuracy')
    
    
    mean_accuracy = cv_scores.mean()
    mean_accuracies.append(mean_accuracy)


best_p_idx = np.argmax(mean_accuracies)
best_p = p_values[best_p_idx]
best_p_accuracy = mean_accuracies[best_p_idx]


print(f'Оптимальне значення p: {best_p:.4f}')
print(f'Максимальна середня точність для p={best_p:.4f}: {best_p_accuracy:.4f}')

plt.plot(p_values, mean_accuracies, marker='o')
plt.xlabel('Значення p')
plt.ylabel('Середня точність на крос-валідації')
plt.title('Середня точність для різних значень p в KNN')
plt.grid()
plt.show()


from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score

centroid_clf = NearestCentroid()
cv_scores_centroid = cross_val_score(centroid_clf, X, y, cv=5, scoring='accuracy')

print(f'Крос-валідаційна точність для NearestCentroid: {cv_scores_centroid}')
print(f'Середня точність на крос-валідації для NearestCentroid: {cv_scores_centroid.mean():.4f}')
print(f'Стандартне відхилення точності для NearestCentroid: {cv_scores_centroid.std():.4f}')
