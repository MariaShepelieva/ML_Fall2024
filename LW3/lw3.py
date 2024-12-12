import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.metrics import  accuracy_score, mean_squared_error, make_scorer, accuracy_score, f1_score

df = pd.read_csv('LW3/bank--additional-full.csv', sep=';')

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


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Точність класифікаційної моделі: {accuracy:.4f}')


reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
print(f'Середньоквадратична похибка регресійної моделі (MSE): {mse:.4f}')


kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
print("Середня точність (accuracy) на крос-валідації:", np.mean(accuracy_scores))


f1_scorer = make_scorer(f1_score)
f1_scores = cross_val_score(clf, X, y, cv=kf, scoring=f1_scorer)
print("Середній F1-score на крос-валідації:", np.mean(f1_scores))

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


p_values = np.linspace(1, 10, 10)
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


#Вибір метрики у методі knn

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

#При необхідності перерахуйте якість за допомогою іншої метрики з списку.

accuracy_scores = []

for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=p, weights='distance')
    scores = cross_val_score(knn, X, y, cv=3, scoring='accuracy', n_jobs=-1) # n_jobs=-1 - використовує всі ядра процесора для прискорення.
    accuracy_scores.append(scores.mean())

# Знаходимо p з найкращою точністю
best_p = p_values[np.argmax(accuracy_scores)]
best_accuracy = max(accuracy_scores)

print("Найкраще значення p:", best_p)
print("Найвища точність (accuracy):", best_accuracy)

plt.plot(p_values, accuracy_scores, marker='o')
plt.xlabel("Значення параметра p")
plt.ylabel("Середня точність (accuracy) на крос-валідації")
plt.title("Залежність точності від параметра p в метриці Мінковського")
plt.show()


from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import cross_val_score

centroid_clf = NearestCentroid()
cv_scores_centroid = cross_val_score(centroid_clf, X, y, cv=5, scoring='accuracy')

print(f'Крос-валідаційна точність для NearestCentroid: {cv_scores_centroid}')
print(f'Середня точність на крос-валідації для NearestCentroid: {cv_scores_centroid.mean():.4f}')
print(f'Стандартне відхилення точності для NearestCentroid: {cv_scores_centroid.std():.4f}')
