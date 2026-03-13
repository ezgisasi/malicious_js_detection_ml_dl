import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV  = 'features.csv'
OUTPUT_CSV = 'features_selected.csv'

df = pd.read_csv(INPUT_CSV)
print(f"✅ Yüklendi: {df.shape}")
print(f"   Benign   : {(df['label']==0).sum():,}")
print(f"   Malicious: {(df['label']==1).sum():,}")

X = df.drop(columns=['filename', 'label'])
y = df['label']
print(f"\n📊 Başlangıç feature sayısı: {X.shape[1]}")

vt      = VarianceThreshold(threshold=0.01)
X_vt    = vt.fit_transform(X)
kept    = X.columns[vt.get_support()].tolist()
removed = [f for f in X.columns if f not in kept]
print(f"\n📉 Adım 1 — VarianceThreshold:")
print(f"   Kalan : {len(kept)}")
print(f"   Elinen: {len(removed)} → {removed}")
X = pd.DataFrame(X_vt, columns=kept)

corr_matrix = X.corr().abs()
upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper.columns if any(upper[col] > 0.95)]
X           = X.drop(columns=to_drop)
print(f"\n📉 Adım 2 — Korelasyon filtresi (|r|>0.95):")
print(f"   Kalan : {X.shape[1]}")
print(f"   Elinen: {len(to_drop)} → {to_drop[:5]}{'...' if len(to_drop)>5 else ''}")

print(f"\n⏳ Adım 3 — Mutual Information hesaplanıyor...")
mi      = mutual_info_classif(X, y, random_state=42)
mi_ser  = pd.Series(mi, index=X.columns).sort_values(ascending=False)
top60   = mi_ser.head(60).index.tolist()
X       = X[top60]
print(f"   Kalan : {X.shape[1]}")

print(f"\n⏳ Adım 4 — Random Forest eğitiliyor...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X, y)
imp    = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top40  = imp.head(40).index.tolist()
X_final = X[top40]
print(f"   Kalan : {X_final.shape[1]}")

df_out           = X_final.copy()
df_out['label']  = y.values
df_out['filename'] = df['filename'].values
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n💾 Kaydedildi: {OUTPUT_CSV}  →  {df_out.shape}")

plt.figure(figsize=(10, 12))
imp.head(40).plot(kind='barh', color='tomato')
plt.xlabel('Feature Importance (Gini)')
plt.title('Top 40 Feature — Random Forest Importance', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 12))
mi_ser.head(40).plot(kind='barh', color='steelblue')
plt.xlabel('Mutual Information Skoru')
plt.title('Top 40 Feature — Mutual Information', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('mi_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"""
{'='*55}
  FEATURE SELECTION TAMAMLANDI ✅
{'='*55}
  Başlangıç : {len(df.columns)-2} feature
  Final     : {len(top40)} feature

  Seçilen featurelar:""")
for i, f in enumerate(top40, 1):
    print(f"  {i:2d}. {f}")
print(f"""
  Dosyalar:
  ✅ features_selected.csv
  ✅ rf_importance.png
  ✅ mi_importance.png
{'='*55}
""")