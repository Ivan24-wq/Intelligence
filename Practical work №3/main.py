import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models, callbacks, losses
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X = np.concatenate([X_train_full, X_test], axis=0)
y = np.concatenate([y_train_full, y_test], axis=0)

# небольшая подвыборка для традиционных методов (иначе SVM будет долго)
SUBSET_FOR_CLASSICAL = 15000  # оставь None для полного набора (но будет медленнее)
if SUBSET_FOR_CLASSICAL:
    idx = np.random.RandomState(42).choice(len(X), size=SUBSET_FOR_CLASSICAL, replace=False)
    X_small = X[idx]
    y_small = y[idx]
else:
    X_small = X
    y_small = y

# нормализация для нейросетей
X_nn = X.astype('float32') / 255.0
X_nn = np.expand_dims(X_nn, -1)  # (N,28,28,1)

# разделение на train/val/test для нейросетей
X_train, X_test_nn, y_train, y_test_nn = train_test_split(X_nn, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)
# итог: ~60% train, 10% val, 30% test

print("Shapes NN:", X_train.shape, X_val.shape, X_test_nn.shape)

# для classical ML: плоский вектор + стандартизация / PCA опционально
X_small_flat = X_small.reshape(len(X_small), -1).astype('float32') / 255.0
scaler = StandardScaler()
X_small_scaled = scaler.fit_transform(X_small_flat)

# PCA для уменьшения размерности (ускорение SVM/RF) — сохраняем 100 компонент
pca = PCA(n_components=100, random_state=42)
X_small_pca = pca.fit_transform(X_small_scaled)
print("Classical data shape (PCA):", X_small_pca.shape)


# 1. RNN / LSTM (изображение -> последовательность)
# Подход: каждую строку 28 пикселей считаем timestep, размерность 28 (features). Последовательность длины 28.
X_train_seq = X_train.squeeze(-1)  # (N,28,28)
X_val_seq = X_val.squeeze(-1)
X_test_seq = X_test_nn.squeeze(-1)

# LSTM требует shape (samples, timesteps, features) => (N,28,28)
def build_lstm_model(hidden_units=64):
    inp = layers.Input(shape=(28, 28))
    x = layers.Masking()(inp)  # не обязательно, но оставим
    x = layers.LSTM(hidden_units, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n=> Обучаем LSTM (используем 28 timesteps x 28 features)")
lstm = build_lstm_model(128)
start = time.time()
history_lstm = lstm.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val), epochs=8, batch_size=128, verbose=1)
lstm_time = time.time() - start
print("LSTM training time: %.1f s" % lstm_time)
loss, acc = lstm.evaluate(X_test_seq, y_test_nn, verbose=0)
print("LSTM test acc:", acc)

# 2. Автокодировщик (unsupervised)
# Преобразуем в плоский вектор для простого dense автокодировщика или используем сверточный автокодировщик.
X_ae_train = X_train  # (N,28,28,1)
X_ae_val = X_val
X_ae_test = X_test_nn

def build_conv_autoencoder(latent_dim=64):
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(latent_dim, activation='relu', name='latent')(x)

    # decoder
    x = layers.Dense(7*7*64, activation='relu')(encoded)
    x = layers.Reshape((7,7,64))(x)
    x = layers.Conv2DTranspose(64, (3,3), strides=1, padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=1, padding='same', activation='relu')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

    ae = models.Model(inp, decoded)
    encoder = models.Model(inp, encoded)  # отдельно encoder для получения латентных векторов
    ae.compile(optimizer='adam', loss='mse')
    return ae, encoder

print("\n=> Обучаем сверточный автокодировщик")
ae, encoder = build_conv_autoencoder(latent_dim=64)
ae.fit(X_ae_train, X_ae_train, validation_data=(X_ae_val, X_ae_val), epochs=15, batch_size=256, verbose=1)
# получаем латентные представления для теста
Z_test = encoder.predict(X_ae_test)
print("Latent shape:", Z_test.shape)

# Визуализация примеров реконструкции
n = 6
recon = ae.predict(X_ae_test[:n])
plt.figure(figsize=(10,4))
for i in range(n):
    plt.subplot(2,n,i+1)
    plt.imshow(X_ae_test[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(2,n,n+i+1)
    plt.imshow(recon[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle("Оригинал (верх) vs Реконструкция (низ)")
plt.show()

# 3. Сравнение CNN с традиционными методами
# 3a: CNN (сверточная модель)
def build_cnn_small():
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu')(x)
    out = layers.Dense(10,activation='softmax')(x)
    model = models.Model(inp,out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n=> Обучаем базовый CNN")
cnn = build_cnn_small()
start = time.time()
cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=6, batch_size=128, verbose=1)
cnn_time = time.time() - start
print("CNN training time: %.1f s" % cnn_time)
loss, cnn_acc = cnn.evaluate(X_test_nn, y_test_nn, verbose=0)
print("CNN test acc:", cnn_acc)

# 3b: Традиционные методы (RandomForest, LogisticRegression, SVM) на PCA-признаках
print("\n=> Обучаем классические методы на PCA-данных (subset)")

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_small_pca, y_small, test_size=0.2, random_state=42, stratify=y_small)

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
start = time.time()
rf.fit(Xc_train, yc_train)
rf_time = time.time() - start
rf_acc = rf.score(Xc_test, yc_test)
print("RF time: %.1f s, acc: %.4f" % (rf_time, rf_acc))

# LogisticRegression (multinomial)
lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga', n_jobs=-1)
start = time.time()
lr.fit(Xc_train, yc_train)
lr_time = time.time() - start
lr_acc = lr.score(Xc_test, yc_test)
print("LR time: %.1f s, acc: %.4f" % (lr_time, lr_acc))

# SVM (RBF) — может быть медленным
svc = SVC(kernel='rbf', gamma='scale')
start = time.time()
svc.fit(Xc_train, yc_train)
svc_time = time.time() - start
svc_acc = svc.score(Xc_test, yc_test)
print("SVC time: %.1f s, acc: %.4f" % (svc_time, svc_acc))


# 4. Анализ преимуществ и недостатков (подытожим результаты)
print("\n=== Сводная таблица (точности) ===")
results = {
    'Model': ['LSTM(seq)', 'CNN', 'Autoencoder+Linear', 'RandomForest', 'LogisticRegression', 'SVM-RBF'],
    'Test accuracy': [
        round(float(acc),4),                      # LSTM acc variable 'acc'
        round(float(cnn_acc),4),
        None,  # заполнится ниже: we'll train a linear classifier on AE latents
        round(float(rf_acc),4),
        round(float(lr_acc),4),
        round(float(svc_acc),4)
    ],
    'Train time (s)': [round(lstm_time,1), round(cnn_time,1), None, round(rf_time,1), round(lr_time,1), round(svc_time,1)]
}

# 4a: Автокодировщик + линейный классификатор на латентных признаках
# берем латентные вектора из encoder для подвыборки X_small
print("\n=> Обучаем LogisticRegression на латентных векторах AE (вся тестовая выборка)")
# получим латенты для небольшого поднабора (чтобы не было долго)
sub_idx = np.random.RandomState(1).choice(len(X), size=10000, replace=False)
X_sub = X[sub_idx].astype('float32') / 255.0
X_sub = np.expand_dims(X_sub, -1)
y_sub = y[sub_idx]

Z_sub = encoder.predict(X_sub, batch_size=256)
Z_train, Z_test_lat, yZ_train, yZ_test = train_test_split(Z_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

lr_ae = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga', n_jobs=-1)
start = time.time()
lr_ae.fit(Z_train, yZ_train)
lr_ae_time = time.time() - start
lr_ae_acc = lr_ae.score(Z_test_lat, yZ_test)
print("LogReg on AE-latent: time %.1f s acc %.4f" % (lr_ae_time, lr_ae_acc))

# добавляем в results
results['Test accuracy'][2] = round(float(lr_ae_acc),4)
results['Train time (s)'][2] = round(lr_ae_time,1)

df_results = pd.DataFrame(results)
print(df_results)

# 5. Доп. оценки: confusion matrix и classification report для CNN и RF
print("\n=== Classification report: CNN (test) ===")
y_pred_cnn = np.argmax(cnn.predict(X_test_nn), axis=1)
print(classification_report(y_test_nn, y_pred_cnn))

print("\n=== Confusion matrix: CNN ===")
print(confusion_matrix(y_test_nn, y_pred_cnn))

print("\n=== Classification report: RandomForest (on PCA subset test) ===")
yc_pred_rf = rf.predict(Xc_test)
print(classification_report(yc_test, yc_pred_rf))


# Сохраняем результаты в CSV
df_results.to_csv("comparison_results.csv", index=False)
print("\nSaved results to comparison_results.csv")
