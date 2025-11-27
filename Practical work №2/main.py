# Ваш код для Задания 2
#Библиотеки
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#Код
CSV_FILE = ("train.csv")
df = pd.read_csv(CSV_FILE)  #Загружаем датасет

#Для новых классов
target = "Pclass"

#Удаление ненужных колонок
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

#Кодировка
df = pd.get_dummies(df)
df = df.fillna(df.mean())
X = df.drop(columns=[target])
y = df[target]

#2D Визуализация
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

#Обучение пар классов
def analyze(class_a, class_b, pair_id):
    #Выборка 2 классов
    mask = (y == class_a) | (y == class_b)
    Xa = X2[mask]
    ya = y[mask]
    
    #Перекодировка
    ya_bin = np.where(ya == class_a, 1, -1)
    #Обучение перцептрона
    clf = Perceptron(max_iter=2000, random_state=51)
    clf.fit(Xa, ya_bin)
    y_pred = clf.predict(Xa)
    acc = accuracy_score(ya_bin, y_pred)
    
    #График
    plt.figure(figsize=(6, 5))
    plt.scatter(Xa[:, 0], Xa[:, 1], c = ya_bin, cmap="coolwarm", alpha=0.6)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    #Построение прямой
    x_line = np.linspace(Xa[:, 0].min(), Xa[:, 0].max(), 100)
    y_line = -(w[0] * x_line + b) / w[1]
    
    #График
    plt.plot(x_line, y_line, "k--", label = "Граница")
    plt.title(f"Пара кластеров {class_a} vs {class_b}\nТочность: ")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Пара кластеров {class_a} vs {class_b} === ")
    print(f"Точность: {acc:.4f}")
    print()
    
    return acc

# 1. 1 класс vs 2 класс
acc12 = analyze(1, 2, 1)

# 2. 1 класс vs 3 класс
acc13 = analyze(1, 3, 2)

# 3. 2 класс vs 3 класс
acc23 = analyze(2, 3, 3)

# 4. Анализ результатов
print("Итоговый анализ")
print(f"1 vs 2: {acc12:.4f}")
print(f" vs 3: {acc13:.4f}")
print(f"2 vs 3: {acc23:.4f}")