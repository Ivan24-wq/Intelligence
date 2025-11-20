import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#Загрузка данных
df = pd.read_csv("train.csv")

# Сохраним истинные метки (Survived)
true_labels = df["Survived"]

# Удаляем неинформативные текстовые колонки
df = df.drop(columns=["Name", "Ticket", "Cabin"])

# Кодируем категориальные признаки
df = pd.get_dummies(df, drop_first=True)

# Заполняем пропуски
df = df.fillna(df.mean())

# Удаляем Survived из признаков
X = df.drop(columns=["Survived"])

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Кластеризация
cluster_range = [2, 3, 4]
scores = {}

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    scores[k] = sil
    print(f"K={k}, silhouette_score={sil:.4f}")

#Визуализация кластеров
plt.figure(figsize=(12, 4))

for i, k in enumerate(cluster_range):
    plt.subplot(1, 3, i+1)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis')
    plt.title(f"K={k}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

plt.show()

#Анализ соответствия кластеров и Survived
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

print("\nСравнение распределения кластеров и Survived:")
print(pd.crosstab(cluster_labels, true_labels))

#Анализ характеристик кластеров
df_clustered = df.copy()
df_clustered["cluster"] = cluster_labels

print("\nСредние значения признаков по кластерам:")
print(df_clustered.groupby("cluster").mean())
