import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

#звргузка целевого датасета
df = pd.read_csv("train.csv")
df = df.dropna(subset=["Fare"]) #Удаление стро без цены

#Выборка признаков
X = df[["Pclass", "Age", "SibSp", "Parch"]].copy()

#Заполняем пропуски возраста
X["Age"].fillna(X["Age"].median(), inplace=True)
#Целевая переменная(fare) т.к модель предсказывает стоимость билета
y = df["Fare"]

#Разделение данных на обучающую и тестовую модель в соотношении 70 / 30 соответственно
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)
print("Размер обучающей: ", X_train.shape)
print("Размер тестовой: ", X_test.shape)