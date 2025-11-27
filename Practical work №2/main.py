# Ваш код для Задания 1
#Импорт библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Многоклассовая классификация с помощью встроенного метода
CSV_FILE = "train.csv"
df = pd.read_csv(CSV_FILE)
#Целевая переменная
target_col = "Survived"

#Удаляем ненужные столбцы
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

#разделение признаков
X = df.drop(columns=[target_col])
y = df[target_col]

#Кодировка текстовых признаков
X = pd.get_dummies(X)
#Заполнение пропусков
X = X.fillna(X.mean())

#Обучающая выборка
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 51, stratify = y   #Разделение в соотношении 70 на обучение, 30 на тест
)

clf_mc = Perceptron(max_iter=1000, random_state=51)
clf_mc.fit(X_train, y_train)

y_pred_mc = clf_mc.predict(X_test)

print("\n Встроенный Перцептрон")
print(f"Точность: {accuracy_score(y_test, y_pred_mc)}")
print(classification_report(y_test, y_pred_mc))
# 2. Подход "один против всех" (One-vs-Rest)
unique_classes = np.unique(y_train)
binary_classifier = {}

#Бинарные метки
for cls in unique_classes:
    y_train_bin = np.where(y_train == cls, 1, -1)
    clf = Perceptron(max_iter=1000, random_state=51)
    clf.fit(X_train, y_train_bin)
    binary_classifier[cls] = clf

#Предсказание Ovr
def predict_ovr(X):
    scores = []
    for cls, clf in binary_classifier.items():
        score = X.values @ clf.coef_.T + clf.intercept_
        scores.append(score)
    scores = np.hstack(scores)      # превращаем список массивов в матрицу N x C
    return np.argmax(scores, axis=1)


y_pred_ovr = predict_ovr(X_test)

# 3. Сравнение результатов
print("\n Бинарный перцептрон")
print(f"Точность: {accuracy_score(y_test, y_pred_ovr)}")
print(classification_report(y_test, y_pred_ovr))