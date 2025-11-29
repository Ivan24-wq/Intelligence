import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 1. Загрузка и подготовка данных

df = pd.read_csv("train.csv")

df = df.drop(columns=["Name", "Ticket", "Cabin"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

y = df["Survived"].values
X = df.drop(columns=["Survived", "PassengerId"]).values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 2. Обучение моделей + замер времени

results = {}

#Перцептрон
start = time.time()
perc = Perceptron(eta0=0.1, max_iter=500)
perc.fit(X_train, y_train)
end = time.time()
acc = accuracy_score(y_test, perc.predict(X_test))
results["Перцептрон"] = (acc, end - start)

#Логистическая регрессия
start = time.time()
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
end = time.time()
acc = accuracy_score(y_test, logreg.predict(X_test))
results["Логистическая регрессия"] = (acc, end - start)

#Дерево решений
start = time.time()
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
end = time.time()
acc = accuracy_score(y_test, tree.predict(X_test))
results["Дерево решений"] = (acc, end - start)

# KMeans не знает правильных меток → назначаем кластеры как классы
start = time.time()
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
end = time.time()

cluster_labels = kmeans.predict(X_test)

# кластеры могут быть перевёрнуты → берём лучшее соответствие
acc1 = accuracy_score(y_test, cluster_labels)
acc2 = accuracy_score(y_test, 1 - cluster_labels)
acc = max(acc1, acc2)

results["KMeans (кластеризация)"] = (acc, end - start)



# 3. Вывод таблицы сравнения
print("\n===== Сравнение моделей =====\n")
print(f"{'Модель':<30} {'Accuracy':<12} {'Время (с)':<12}")
print("-" * 60)

for model, (acc, t) in results.items():
    print(f"{model:<30} {acc:<12.4f} {t:<12.4f}")


# 4. Анализ: где перцептрон лучше?
print("\n\n===== Анализ результатов =====")

acc_p = results["Перцептрон"][0]
acc_log = results["Логистическая регрессия"][0]
acc_tree = results["Дерево решений"][0]
acc_km = results["KMeans (кластеризация)"][0]

print(f"\nПерцептрон accuracy = {acc_p:.4f}")

print("\n► Перцептрон показывает лучшие результаты, когда:")
print("- данные примерно линейно разделимы")
print("- признаки нормированы (как мы сделали StandardScaler)")
print("- нет слишком сильных выбросов")
print("- классы сбалансированы")

print("\n► На Titanic перцептрон обычно хуже логистической регрессии, потому что:")
print("- реальные зависимости нелинейны")
print("- важны взаимодействия признаков (дерево их ловит лучше)")
print("- перцептрон улавливает только линейную границу")

print("\n► Почему логистическая регрессия часто лучше:")
print("- использует вероятностную модель")
print("- стабильнее в обучении")
print("- использует сглаженный градиент")

print("\n► Почему дерево часто даёт хорошую точность:")
print("- ловит нелинейности")
"- учитывает сложные взаимодействия между признаками"

print("\n► Почему кластеризация слабая:")
print("- KMeans не использует классы"
"- он пытается разбить данные без информации о выживании"
"- поэтому качество ниже, чем у моделей классификации")
