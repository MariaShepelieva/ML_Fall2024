import numpy as np
import pandas as pd
import os

file_path = './vodafone_age_subset.csv'
data = pd.read_csv(file_path, delimiter=',')

# Форма датасета
n_rows, n_cols = data.shape
print(f"Кількість рядків: {n_rows}")
print(f"Кількість колонок: {n_cols}\n")

# Перевірка відсутніх значень
missing_values = data.isnull().sum()
print("Відсутні значення:")
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Жодна колонка не має пропущених значень.\n")

# Виявлення постійних колонок
constant_columns = [col for col in data.columns if data[col].nunique() == 1]
print(f"Постійні колонки: {constant_columns}\n")

# Перевірка високої кардинальності
high_cardinality_columns = [col for col in data.columns if data[col].nunique() > 500]
print(f"Колонки з високою кардинальністю: {high_cardinality_columns}\n")

# Перевірка дублікатів
duplicates = data.duplicated().sum()
print(f"Кількість дубльованих рядків: {duplicates}\n")

# Приклад перших 5 рядків
print("Приклад даних:")
print(data.head())

# Розподіл колонок за типами
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
identification_columns = [col for col in data.columns if 'id' in col.lower() or 'hash' in col.lower()]

print(f"Категоріальні дані\n: {categorical_columns}")
print(f"Числові дані\n: {numerical_columns}")
print(f"Ідентифікаційні дані\n: {identification_columns}")


print(data.info())
print(data.describe().T)

data = data.drop(constant_columns, axis=1)
data = data.drop(high_cardinality_columns, axis=1)
print(f"Оновлена кількість колонок: {data.shape[1]}")
print(data.info())
print(data.describe().T)

print('Кількість унікальних значень\n',data.target.value_counts())