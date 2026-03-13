import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings, os, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                      Flatten, LSTM, Bidirectional, GRU,
                                      BatchNormalization, Reshape)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("  JS Kötücül Kod Tespiti — 12 Model Karşılaştırması")
print("=" * 60)

print("\n📂 Veri yükleniyor...")
df = pd.read_csv('features_selected.csv')

drop_cols = [c for c in ['filename', 'label'] if c in df.columns]
X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
y = df['label']

print(f"  Toplam örnek  : {len(df):,}")
print(f"  Benign        : {(y==0).sum():,}")
print(f"  Malicious     : {(y==1).sum():,}")
print(f"  Feature sayısı: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_features = X_train_sc.shape[1]

cw      = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n  Sınıf ağırlıkları → Benign: {cw[0]:.3f}  |  Malicious: {cw[1]:.3f}")
print(f"  (tüm modellerde class_weight='balanced' uygulandı ✅)")

def evaluate(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    print(f"  ✅ {name:<25} Acc:{acc:.4f}  P:{prec:.4f}  R:{rec:.4f}  F1:{f1:.4f}  AUC:{auc:.4f}")
    return {'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'ROC-AUC': auc}

results       = []
conf_matrices = {}

print("\n" + "─"*60)
print("  📊 GELENEKSEL MAKİNE ÖĞRENMESİ MODELLERİ")
print("─"*60)

print("\n[1/12] Logistic Regression eğitiliyor...")
t  = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                         random_state=42, n_jobs=-1)
lr.fit(X_train_sc, y_train)
y_pred = lr.predict(X_test_sc)
y_prob = lr.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("Logistic Regression", y_test, y_pred, y_prob))
conf_matrices["Logistic Regression"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[2/12] Naive Bayes eğitiliyor...")
t  = time.time()
nb = GaussianNB()
nb.fit(X_train_sc, y_train)
y_pred = nb.predict(X_test_sc)
y_prob = nb.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("Naive Bayes", y_test, y_pred, y_prob))
conf_matrices["Naive Bayes"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[3/12] KNN eğitiliyor... (biraz sürebilir)")
t   = time.time()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_sc, y_train)
y_pred = knn.predict(X_test_sc)
y_prob = knn.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("KNN", y_test, y_pred, y_prob))
conf_matrices["KNN"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[4/12] SVM eğitiliyor... (en uzun süren ML modeli)")
t   = time.time()
svm = SVC(kernel='rbf', class_weight='balanced',
           probability=True, random_state=42)
svm.fit(X_train_sc, y_train)
y_pred = svm.predict(X_test_sc)
y_prob = svm.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("SVM", y_test, y_pred, y_prob))
conf_matrices["SVM"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[5/12] Decision Tree eğitiliyor...")
t  = time.time()
dt = DecisionTreeClassifier(class_weight='balanced',
                              random_state=42, max_depth=20)
dt.fit(X_train_sc, y_train)
y_pred = dt.predict(X_test_sc)
y_prob = dt.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("Decision Tree", y_test, y_pred, y_prob))
conf_matrices["Decision Tree"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[6/12] Random Forest eğitiliyor...")
t  = time.time()
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                              random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)
y_prob = rf.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("Random Forest", y_test, y_pred, y_prob))
conf_matrices["Random Forest"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n[7/12] XGBoost eğitiliyor...")
t   = time.time()
xgb = XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos,
                     random_state=42, n_jobs=-1,
                     eval_metric='logloss', verbosity=0)
xgb.fit(X_train_sc, y_train)
y_pred = xgb.predict(X_test_sc)
y_prob = xgb.predict_proba(X_test_sc)[:, 1]
results.append(evaluate("XGBoost", y_test, y_pred, y_prob))
conf_matrices["XGBoost"] = confusion_matrix(y_test, y_pred)
print(f"     Süre: {time.time()-t:.1f}s")

print("\n" + "─"*60)
print("  🧠 DERİN ÖĞRENME MODELLERİ")
print("─"*60)

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

def dl_evaluate(name, model, X_tr, X_te, y_tr, y_te):
    t = time.time()
    history = model.fit(
        X_tr, y_tr,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[es],
        class_weight=cw_dict,  
        verbose=0
    )
    y_prob = model.predict(X_te, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"     Süre: {time.time()-t:.1f}s  |  Epoch: {len(history.history['loss'])}")
    r = evaluate(name, y_te, y_pred, y_prob)
    conf_matrices[name] = confusion_matrix(y_te, y_pred)
    return r

print("\n[8/12] MLP eğitiliyor...")
mlp = Sequential([
    Dense(256, activation='relu', input_shape=(n_features,)),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(), Dropout(0.3),
    Dense(64,  activation='relu'), Dropout(0.2),
    Dense(1,   activation='sigmoid')
])
mlp.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
results.append(dl_evaluate("MLP", mlp, X_train_sc, X_test_sc, y_train, y_test))

print("\n[9/12] CNN eğitiliyor...")
cnn = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Conv1D(64,  3, activation='relu', padding='same'),
    BatchNormalization(), MaxPooling1D(2),
    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(), MaxPooling1D(2),
    Conv1D(64,  3, activation='relu', padding='same'),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(1,   activation='sigmoid')
])
cnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
results.append(dl_evaluate("CNN", cnn, X_train_sc, X_test_sc, y_train, y_test))

print("\n[10/12] LSTM eğitiliyor...")
lstm_model = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    LSTM(128, return_sequences=True), Dropout(0.3),
    LSTM(64),                         Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
results.append(dl_evaluate("LSTM", lstm_model, X_train_sc, X_test_sc, y_train, y_test))

print("\n[11/12] BiLSTM eğitiliyor...")
bilstm = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    Bidirectional(LSTM(128, return_sequences=True)), Dropout(0.3),
    Bidirectional(LSTM(64)),                         Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
bilstm.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
results.append(dl_evaluate("BiLSTM", bilstm, X_train_sc, X_test_sc, y_train, y_test))

print("\n[12/12] GRU eğitiliyor...")
gru_model = Sequential([
    Reshape((n_features, 1), input_shape=(n_features,)),
    GRU(128, return_sequences=True), Dropout(0.3),
    GRU(64),                         Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
gru_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
results.append(dl_evaluate("GRU", gru_model, X_train_sc, X_test_sc, y_train, y_test))

df_results = pd.DataFrame(results).sort_values('F1', ascending=False)
df_results.to_csv('results/model_sonuclari.csv', index=False)

print("\n" + "="*60)
print("  📊 SONUÇ TABLOSU (F1'e göre sıralı)")
print("="*60)
print(df_results.to_string(index=False, float_format='{:.4f}'.format))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
colors  = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6',
           '#1abc9c','#e67e22','#e91e63','#00bcd4','#8bc34a','#ff5722','#607d8b']
ML_MODELS = ['Logistic Regression','Naive Bayes','KNN','SVM',
             'Decision Tree','Random Forest','XGBoost']

fig, axes = plt.subplots(1, 5, figsize=(24, 7))
for ax, metric in zip(axes, metrics):
    vals  = df_results[metric].values
    names = df_results['Model'].values
    bars  = ax.barh(names, vals, color=colors[:len(names)], edgecolor='white')
    ax.set_xlim(0, 1.05)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Skor')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5)
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
plt.suptitle('12 Model Karşılaştırması — Tüm Metrikler', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/model_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✅ model_karsilastirma.png")

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = ['#2980b9' if m in ML_MODELS else '#c0392b'
              for m in df_results['Model']]
bars = ax.barh(df_results['Model'], df_results['F1'],
               color=bar_colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, df_results['F1']):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlim(0, 1.08)
ax.set_xlabel('F1 Skoru', fontsize=12)
ax.set_title('Model Karşılaştırması — F1 Skoru\n(Mavi: ML  |  Kırmızı: DL)',
             fontsize=13, fontweight='bold')
ax.axvline(x=0.9,  color='gray',   linestyle='--', alpha=0.5)
ax.axvline(x=0.95, color='orange', linestyle='--', alpha=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#2980b9', label='Geleneksel ML'),
                   Patch(color='#c0392b', label='Derin Öğrenme')], fontsize=10)
plt.tight_layout()
plt.savefig('results/f1_karsilastirma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ f1_karsilastirma.png")

best4 = df_results.head(4)['Model'].tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, best4):
    cm = conf_matrices[name]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign','Malicious'],
                yticklabels=['Benign','Malicious'])
    f1_val = df_results[df_results['Model']==name]['F1'].values[0]
    ax.set_title(f'{name}\nF1={f1_val:.4f}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gerçek')
    ax.set_xlabel('Tahmin')
plt.suptitle('En İyi 4 Model — Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ confusion_matrix.png")

ml_df  = df_results[df_results['Model'].isin(ML_MODELS)]
dl_df  = df_results[~df_results['Model'].isin(ML_MODELS)]
ml_avg = ml_df[metrics].mean()
dl_avg = dl_df[metrics].mean()

fig, ax = plt.subplots(figsize=(10, 6))
x, w = np.arange(len(metrics)), 0.35
ax.bar(x - w/2, ml_avg, w, label='ML Ortalaması', color='#2980b9', alpha=0.85)
ax.bar(x + w/2, dl_avg, w, label='DL Ortalaması', color='#c0392b', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Ortalama Skor', fontsize=12)
ax.set_title('Geleneksel ML vs Derin Öğrenme — Ortalama Performans',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
for i, (mv, dv) in enumerate(zip(ml_avg, dl_avg)):
    ax.text(i - w/2, mv + 0.01, f'{mv:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, dv + 0.01, f'{dv:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('results/ml_vs_dl.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ ml_vs_dl.png")

best = df_results.iloc[0]
print(f"""
{'='*60}
  ✅ MODEL EĞİTİMİ TAMAMLANDI!
{'='*60}
  En iyi model : {best['Model']}
  F1 Skoru     : {best['F1']:.4f}
  Accuracy     : {best['Accuracy']:.4f}
  ROC-AUC      : {best['ROC-AUC']:.4f}

  ML Ortalama F1 : {ml_df['F1'].mean():.4f}
  DL Ortalama F1 : {dl_df['F1'].mean():.4f}

  Çıktılar (results/ klasörü):
  ✅ model_sonuclari.csv
  ✅ model_karsilastirma.png
  ✅ f1_karsilastirma.png
  ✅ confusion_matrix.png
  ✅ ml_vs_dl.png
{'='*60}
""")
