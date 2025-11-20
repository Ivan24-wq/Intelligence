import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

#Подготовка данных
df = pd.read_csv("train.csv")

# Удаляем текстовые колонки
df = df.drop(columns=["Name", "Ticket", "Cabin"])

# Кодируем категориальные признаки
df = pd.get_dummies(df, drop_first=True)

# Заполняем пропуски
df = df.fillna(df.mean())

# Разделяем на признаки и целевую переменную
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#VotingClassifier
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
print("VotingClassifier Accuracy:", accuracy_score(y_test, y_pred_voting))

#BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

bagging_clf.fit(X_train, y_train)
y_pred_bag = bagging_clf.predict(X_test)
print("BaggingClassifier Accuracy:", accuracy_score(y_test, y_pred_bag))


#GradientBoostingClassifier + GridSearch
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

gb_clf = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_gb = grid_search.best_estimator_
y_pred_gb = best_gb.predict(X_test)

print("Best GridSearch GradientBoosting params:", grid_search.best_params_)
print("GradientBoosting Accuracy:", accuracy_score(y_test, y_pred_gb))

#Сравнение всех моделей

results = {
    'VotingClassifier': accuracy_score(y_test, y_pred_voting),
    'BaggingClassifier': accuracy_score(y_test, y_pred_bag),
    'GradientBoosting': accuracy_score(y_test, y_pred_gb)
}

print("\nAccuracy comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")


#RandomizedSearchCV для GradientBoosting
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

rand_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42
)

rand_search.fit(X_train, y_train)
best_rand_gb = rand_search.best_estimator_
print("\nBest RandomizedSearchCV GradientBoosting params:", rand_search.best_params_)
print("Accuracy after RandomizedSearchCV:", accuracy_score(y_test, best_rand_gb.predict(X_test)))

#Кривые обучения
train_sizes, train_scores, test_scores = learning_curve(
    best_rand_gb, X_train, y_train,
    cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train score')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Validation score')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve - GradientBoosting')
plt.legend()
plt.grid(True)
plt.show()
