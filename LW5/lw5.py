import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score
import os

# Directory Setup
results_dir = r'D:\Machine Learning Course\ML_Fall2024\LW5\Results'
os.makedirs(results_dir, exist_ok=True)

# Load dataset
df = pd.read_csv('./bank--additional-full.csv', sep=";", na_values='unknown')

# Initial Data Analysis
print("\u041f\u0435\u0440\u0448\u0456 \u0440\u044f\u0434\u043a\u0438 \u0434\u0430\u043d\u0438\u0445:")
print(df.head())
print("\n\u0406\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0456\u044f \u043f\u0440\u043e \u0434\u0430\u0442\u0430\u0444\u0440\u0435\u0439\u043c:")
print(df.info())
print("\n\u041a\u0456\u043b\u044c\u043a\u0456\u0441\u0442\u044c \u043f\u0440\u043e\u043f\u0443\u0449\u0435\u043d\u0438\u0445 \u0437\u043d\u0430\u0447\u0435\u043d\u044c \u0443 \u043a\u043e\u0436\u043d\u043e\u043c\u0443 \u0441\u0442\u043e\u0432\u043f\u0446\u0456:")
print(df.isna().sum())

# Fill missing values
na_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan']
for col in na_columns:
    df[col] = df[col].fillna(df[col].mode().values[0])

print("\n\u041a\u0456\u043b\u044c\u043a\u0456\u0441\u0442\u044c \u043f\u0440\u043e\u043f\u0443\u0449\u0435\u043d\u0438\u0445 \u0437\u043d\u0430\u0447\u0435\u043d\u044c \u043f\u0456\u0441\u043b\u044f \u043e\u0431\u0440\u043e\u0431\u043a\u0438:")
print(df.isna().sum())

# Remove outliers
q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3 - q1
df = df[(df['age'] > q1 - 1.5 * iqr) & (df['age'] < q3 + 1.5 * iqr)]

# Encode categorical variables
label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])
for column in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
    df[column] = label_encoder.fit_transform(df[column])

# Split dataset
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Visualize Decision Tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['no', 'yes'],
    filled=True
)
output_path = os.path.join(results_dir, 'Tree.png')
plt.savefig(output_path)
plt.close()

# Model evaluation function
def select_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return model, y_pred, score, f1, precision

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfc, y_pred, score, f1, precision = select_model(rfc, X_train, y_train, X_test, y_test)

print(f'Accuracy: {score:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')

# Save confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
output_path = os.path.join(results_dir, 'Confusion-Matrix.png')
plt.savefig(output_path)
plt.close()

# Validation Curves
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
    plt.ylabel("F1-score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

n_estimators_range = [50, 100, 200, 300, 400]
train_scores_n, test_scores_n = validation_curve(
    GradientBoostingClassifier(learning_rate=0.1, random_state=42),
    X_train, y_train,
    param_name="n_estimators",
    param_range=n_estimators_range,
    scoring="f1",
    cv=5
)

plot_validation_curve(n_estimators_range, train_scores_n, test_scores_n, "n_estimators", "Validation Curve for GradientBoostingClassifier")

# Final Gradient Boosting Model
final_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=1, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['no', 'yes']))
print("ROC-AUC:", roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1]))

