import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import learning_curve

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


#Оценка качества моделей с помощью MSE, RMSE, MAE, R²
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test, pred)
    R2 = r2_score(y_test, pred)


print(f"\n{name}: ")
print(f"MSE: {MSE}")
print(f"RMSE: {MSE:.4f}")
print(f"MAE: {MAE:.4f}")
print(f"R2: {R2:.4f}")


#Визуализация предсказаний
best_res = fitted_models["RandomForestRegressor"]
pred = best_res.predict(X_test)
plt.scatter(y_test, pred)
plt.xlabel("Реальна] стоимость")
plt.ylabel("Предсказанная стоимость")
plt.title("предсказания vs реальные значения")
plt.show()

#Анализ важности признаков
feature_names = X.columns
importance = best_res.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

print(fi)

#Прверка на переобучение
train_size, train_score, val_score = learning_curve(
    best_res,
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv = 10, #оценка подвыборных данных
    scoring = "neg_mean_squared_error"
)

train_scores_mean = -train_score.mean(axis=1)
val_scores_mean = -val_score.mean(axis=1)

plt.plot(train_size, train_scores_mean, label="Ошибка обучения")
plt.plot(train_size, val_scores_mean, label="Ошибка валидации")
plt.xlabel("Размер обучения")
plt.ylabel("MSE")
plt.legend()
plt.title("Проверка переобучения")
plt.show()