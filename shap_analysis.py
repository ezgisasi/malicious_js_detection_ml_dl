import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

INPUT_CSV    = 'features_selected.csv'
OUTPUT_DIR   = 'shap_outputs'
TOP_N        = 10
SAMPLE_SIZE  = 1000
KERNEL_BG    = 100
KERNEL_EVAL  = 200
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'Random Forest'      : '#E05C3A',
    'XGBoost'            : '#2196F3',
    'Decision Tree'      : '#FF9800',
    'KNN'                : '#9C27B0',
    'Logistic Regression': '#4CAF50',
    'SVM'                : '#00BCD4',
    'Naive Bayes'        : '#795548',
}

def plot_bar(shap_series, model_name, filename, color):
    top = shap_series.head(TOP_N)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=color, alpha=0.85, height=0.6)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + top.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left', fontsize=9)
    ax.set_xlabel('Ortalama |SHAP Değeri|', fontsize=11)
    ax.set_title(f'{model_name}\nSHAP Özellik Önemi (Top {TOP_N})',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0, top.values.max() * 1.18)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   ✅ {path}')

def extract_class1(sv_raw, n_features):
    if isinstance(sv_raw, list):
        arr = np.array(sv_raw[1])
    else:
        arr = np.array(sv_raw)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    if arr.ndim == 2 and arr.shape[1] != n_features:
        arr = arr.T
    return arr

def mean_abs(sv, feat_names):
    vals = np.abs(sv).mean(axis=0)
    return pd.Series(vals, index=feat_names).sort_values(ascending=False)

print('\n' + '='*55)
print('  SHAP ANALİZİ BAŞLIYOR')
print('='*55)

print('\n📂 Veri yükleniyor...')
df    = pd.read_csv(INPUT_CSV)
X     = df.drop(columns=['label', 'filename'], errors='ignore')
y     = df['label']
feats = X.columns.tolist()
print(f'   {df.shape}  |  Benign: {(y==0).sum():,}  |  Malicious: {(y==1).sum():,}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
Xtr    = pd.DataFrame(scaler.fit_transform(X_train), columns=feats)
Xte    = pd.DataFrame(scaler.transform(X_test),      columns=feats)

spw    = (y_train == 0).sum() / (y_train == 1).sum()

rng    = np.random.default_rng(RANDOM_STATE)
idx    = rng.choice(len(Xte), size=min(SAMPLE_SIZE, len(Xte)), replace=False)
X_shap = Xte.iloc[idx]
bg_data = shap.sample(Xtr, KERNEL_BG, random_state=RANDOM_STATE)
X_eval  = X_shap.iloc[:KERNEL_EVAL]

print('\n🏋️  Modeller eğitiliyor...')
models = {
    'Random Forest'      : RandomForestClassifier(n_estimators=200, class_weight='balanced',
                               random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost'            : XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=spw, use_label_encoder=False,
                               eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
    'Decision Tree'      : DecisionTreeClassifier(criterion='gini', max_depth=20,
                               class_weight='balanced', random_state=RANDOM_STATE),
    'KNN'                : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced',
                               random_state=RANDOM_STATE),
    'SVM'                : SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                               probability=True, random_state=RANDOM_STATE),
    'Naive Bayes'        : GaussianNB(),
}

for name, model in models.items():
    print(f'   ⏳ {name}...', end=' ', flush=True)
    model.fit(Xtr, y_train)
    print(f'Accuracy: {model.score(Xte, y_test):.4f}')

rankings = {}
file_map = {
    'Random Forest'      : '01_rf_shap_bar.png',
    'XGBoost'            : '02_xgb_shap_bar.png',
    'Decision Tree'      : '03_dt_shap_bar.png',
    'KNN'                : '04_knn_shap_bar.png',
    'Logistic Regression': '05_lr_shap_bar.png',
    'SVM'                : '06_svm_shap_bar.png',
    'Naive Bayes'        : '07_nb_shap_bar.png',
}

for mname in ['Random Forest', 'XGBoost', 'Decision Tree']:
    print(f'\n🌲 {mname} SHAP (TreeExplainer)...')
    exp = shap.TreeExplainer(models[mname])
    sv  = extract_class1(exp.shap_values(X_shap), len(feats))
    ser = mean_abs(sv, feats)
    rankings[mname] = ser
    plot_bar(ser, mname, file_map[mname], COLORS[mname])

for mname in ['KNN', 'Logistic Regression', 'SVM', 'Naive Bayes']:
    print(f'\n🔍 {mname} SHAP (KernelExplainer)...')
    exp    = shap.KernelExplainer(models[mname].predict_proba, bg_data)
    sv_raw = exp.shap_values(X_eval, nsamples=100)
    sv     = extract_class1(sv_raw, len(feats))
    ser    = mean_abs(sv, feats)
    rankings[mname] = ser
    plot_bar(ser, mname, file_map[mname], COLORS[mname])

print('\n📊 Karşılaştırma grafiği hazırlanıyor...')

ref_feats  = rankings['Random Forest'].head(TOP_N).index.tolist()
model_list = list(rankings.keys())
n_models   = len(model_list)
x          = np.arange(TOP_N)
bw         = 0.10

fig, ax = plt.subplots(figsize=(14, 6))

for i, mname in enumerate(model_list):
    ser    = rankings[mname]
    vals   = np.array([ser.get(f, 0) for f in ref_feats])
    norm   = vals / vals.max() if vals.max() > 0 else vals
    offset = (i - n_models / 2 + 0.5) * bw
    ax.bar(x + offset, norm, bw * 0.92,
           color=COLORS[mname], alpha=0.85, label=mname)

ax.set_xticks(x)
ax.set_xticklabels(ref_feats, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Normalize SHAP Önemi (0–1)', fontsize=11)
ax.set_title('7 ML Modeli — SHAP Özellik Önemi Karşılaştırması (Top 10)',
             fontsize=13, fontweight='bold', pad=14)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
ax.set_ylim(0, 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
cmp_path = os.path.join(OUTPUT_DIR, '08_tum_modeller_karsilastirma.png')
plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'   ✅ {cmp_path}')

csv_df             = pd.DataFrame({m: rankings[m] for m in model_list})
csv_df.index.name  = 'Feature'
csv_path           = os.path.join(OUTPUT_DIR, 'shap_feature_rankings.csv')
csv_df.to_csv(csv_path)
print(f'   ✅ {csv_path}')

print('\n' + '='*55)
print('  SHAP ANALİZİ TAMAMLANDI ✅')
print('='*55)
print(f'\n  {"Model":<22} | Top 3 SHAP Özelliği')
print('  ' + '-'*52)
for mname, ser in rankings.items():
    top3 = ', '.join(ser.head(3).index.tolist())
    print(f'  {mname:<22} | {top3}')
print('\n  Üretilen dosyalar:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'    📄 {f}')
print('='*55 + '\n')
