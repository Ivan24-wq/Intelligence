import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# 1. ПОДГРУЗКА ДАННЫХ
data = pd.read_csv("train.csv")

print(data.head())


# Удаляем ненужные признаки
data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

# Заполняем пропуски
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Label Encoding: Sex
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# One-Hot Encoding: Embarked
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)


# 2. EDA (распределение цели)
sns.countplot(x='Survived', data=data)
plt.title("Распределение целевой переменной")
plt.show()

print(data.info())


# 3. TRAIN/TEST SPLIT

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# 4. ОБУЧЕНИЕ МОДЕЛЕЙ
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=42)
svc = SVC(probability=True, random_state=42)

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
svc.fit(X_train, y_train)


# 5. ОЦЕНКА МОДЕЛИ
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


print("=== KNN ===")
evaluate(knn, X_test, y_test)

print("=== Random Forest ===")
evaluate(rf, X_test, y_test)

print("=== SVC ===")
evaluate(svc, X_test, y_test)


# 6. КРОСС-ВАЛИДАЦИЯ (k=N)
N = 5  # номер студента

models = {"KNN": knn, "RandomForest": rf, "SVC": svc}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=N, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")


# 7. GRIDSEARCHCV ДЛЯ RandomForest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=N,
    scoring='accuracy',
    n_jobs=-1
)

gs.fit(X_train, y_train)

print("\nЛучшие параметры:", gs.best_params_)
print("Лучший score:", gs.best_score_)
