import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

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

#ОБУЧЕНИЕ LinearRegression, DecisionTreeRegressor, RandomForestRegressor

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state = 50),
    "RandomForestRegressor": RandomForestRegressor(random_state = 50)
    
}
fitted_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model


#Оценить качество моделей с помощью MSE, RMSE, MAE, R²
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, pred)
    R2 = r2_score(y_test, pred)

print(f"\n{name}: ")
print(f"{MSE}: ")
print(f"RMSE: {MSE:.4f}")
print(f"MAE: {MAE:.4f}")
print(f"R2: {R2:.4f}")