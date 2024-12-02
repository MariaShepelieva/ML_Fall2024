import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score

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

# Візуалізація дерева рішень
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=df.drop('y', axis=1).columns,
    class_names=['no', 'yes'],  # Відповідає закодованим значенням
    filled=True
)
plt.savefig('ML_Fall2024\LW5\Results\Tree.png')
plt.close()


def select_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return model, y_pred, score, f1, precision



# Оцінка моделі RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)  # Більше дерев для кращої узагальненості
model_rfc, y_pred, score, f1, precision = select_model(rfc, X_train, y_train, X_test, y_test)

print(f'Accuracy: {score:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')

print("\nClassification Report1:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))

print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('ML_Fall2024\LW5\Results\Confusion-Matrix.png')
plt.close()

base_model = DecisionTreeClassifier(max_depth=3)

n_estimators_range = [10, 50, 100, 200, 300, 500]
learning_rate_range = [0.01, 0.1, 0.5, 1, 1.5, 2]

train_scores_n, test_scores_n = validation_curve(
    AdaBoostClassifier(estimator=base_model, learning_rate=1, algorithm='SAMME', random_state=42),
    X_train, y_train,
    param_name="n_estimators",
    param_range=n_estimators_range,
    scoring="f1",
    cv=5
)

train_scores_lr, test_scores_lr = validation_curve(
    AdaBoostClassifier(estimator=base_model, n_estimators=100, algorithm='SAMME', random_state=42),
    X_train, y_train,
    param_name="learning_rate",
    param_range=learning_rate_range,
    scoring="f1",
    cv=5
)

def plot_validation_curve(param_range, train_scores, test_scores, param_name, title):
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
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

plot_validation_curve(n_estimators_range, train_scores_n, test_scores_n, "n_estimators", "Validation Curve for AdaBoost (n_estimators)")
plot_validation_curve(learning_rate_range, train_scores_lr, test_scores_lr, "learning_rate", "Validation Curve for AdaBoost (learning_rate)")

''' Отримали графіки, які допомогли обрати найкращі значення для n_estimators: 200-300. learning_rate: 0.5.'''

final_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), 
                                 n_estimators=250, 
                                 learning_rate=0.5, 
                                 random_state=42)
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)
print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print(classification_report(y_test, y_test_pred, target_names=['no', 'yes']))
print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
print("ROC-AUC:", roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1]))


base_model = DecisionTreeClassifier(max_depth=1)  # Базова модель "неглибоке дерево рішень"

n_estimators_range = [50, 100, 200, 300, 400]
train_scores_n, test_scores_n = validation_curve(
    GradientBoostingClassifier(learning_rate=0.1, random_state=42),
    X_train, y_train,
    param_name="n_estimators",
    param_range=n_estimators_range,
    scoring="f1",  
)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, np.mean(train_scores_n, axis=1), label="Training F1-score", color="blue", marker="o")
plt.plot(n_estimators_range, np.mean(test_scores_n, axis=1), label="Validation F1-score", color="orange", marker="o")
plt.fill_between(
    n_estimators_range,
    np.mean(train_scores_n, axis=1) - np.std(train_scores_n, axis=1),
    np.mean(train_scores_n, axis=1) + np.std(train_scores_n, axis=1),
    alpha=0.2, color="blue"
)
plt.fill_between(
    n_estimators_range,
    np.mean(test_scores_n, axis=1) - np.std(test_scores_n, axis=1),
    np.mean(test_scores_n, axis=1) + np.std(test_scores_n, axis=1),
    alpha=0.2, color="orange"
)
plt.title("Validation Curve for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("F1-score")
plt.legend(loc="best")
plt.grid()
plt.savefig('ML_Fall2024\LW5\Results\Validation Curve for n_estimator.png')
plt.close()

learning_rate_range = [0.01, 0.05, 0.1, 0.2, 0.3]
train_scores_lr, test_scores_lr = validation_curve(
    GradientBoostingClassifier(n_estimators=200, random_state=42),
    X_train, y_train,
    param_name="learning_rate",
    param_range=learning_rate_range,
    scoring="f1",
    cv=5
)

plt.figure(figsize=(10, 6))
plt.plot(learning_rate_range, np.mean(train_scores_lr, axis=1), label="Training F1-score", color="blue", marker="o")
plt.plot(learning_rate_range, np.mean(test_scores_lr, axis=1), label="Validation F1-score", color="orange", marker="o")
plt.fill_between(
    learning_rate_range,
    np.mean(train_scores_lr, axis=1) - np.std(train_scores_lr, axis=1),
    np.mean(train_scores_lr, axis=1) + np.std(train_scores_lr, axis=1),
    alpha=0.2, color="blue"
)
plt.fill_between(
    learning_rate_range,
    np.mean(test_scores_lr, axis=1) - np.std(test_scores_lr, axis=1),
    np.mean(test_scores_lr, axis=1) + np.std(test_scores_lr, axis=1),
    alpha=0.2, color="orange"
)
plt.title("Validation Curve for learning_rate")
plt.xlabel("learning_rate")
plt.ylabel("F1-score")
plt.legend(loc="best")
plt.grid()
plt.savefig('ML_Fall2024\LW5\Results\Validation-Curve-for-learning_rate.png')
plt.close()


best_n_estimators = 200
best_learning_rate = 0.1


final_model = GradientBoostingClassifier(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    max_depth=1, 
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
print("\nClassification Report2:")
print("\nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
print("ROC-AUC:", roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1]))

print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('ML_Fall2024\LW5\Results\Confusion Matrix.png')
plt.close()
