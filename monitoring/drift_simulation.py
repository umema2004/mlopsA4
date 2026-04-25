import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TASK 7: TIME-BASED DRIFT SIMULATION")
print("="*60)

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv('/home/hamza/A4/ieee-fraud-detection/preprocessed.csv')
print(f"\n✅ Loaded data: {df.shape}")
print(f"TransactionDT range: {df['TransactionDT'].min()} → {df['TransactionDT'].max()}")

# ── Split by time (early vs late transactions) ───────────────────
dt_33 = df['TransactionDT'].quantile(0.33)
dt_66 = df['TransactionDT'].quantile(0.66)

train_df = df[df['TransactionDT'] <= dt_33].copy()
mid_df   = df[(df['TransactionDT'] > dt_33) & (df['TransactionDT'] <= dt_66)].copy()
test_df  = df[df['TransactionDT'] > dt_66].copy()

print(f"\n📅 Time-based split:")
print(f"  Train (early 33%):  {len(train_df):,} rows | Fraud: {train_df['isFraud'].mean():.3f}")
print(f"  Mid   (middle 33%): {len(mid_df):,} rows  | Fraud: {mid_df['isFraud'].mean():.3f}")
print(f"  Test  (late 33%):   {len(test_df):,} rows  | Fraud: {test_df['isFraud'].mean():.3f}")

# ── Prepare features ─────────────────────────────────────────────
TARGET = 'isFraud'
DROP_COLS = [TARGET, 'TransactionDT']
feature_cols = [c for c in df.columns if c not in DROP_COLS]

# Apply feature selector
    fs = joblib.load("/home/hamza/A4/models/feature_selector.pkl")
X_train_raw = train_df[feature_cols].fillna(0)
X_train = pd.DataFrame(fs.transform(X_train_raw))
X_test_raw  = test_df[feature_cols].fillna(0)
X_test  = pd.DataFrame(fs.transform(X_test_raw))
y_train = train_df[TARGET]
X_test  = test_df[feature_cols].fillna(0)
y_test  = test_df[TARGET]

print(f"\n✅ Features: {len(feature_cols)}")

# ── Load existing model (trained on all data) ────────────────────
model = joblib.load('/home/hamza/A4/models/hybrid_model.pkl')
print("✅ Loaded hybrid model")

# ── Evaluate on EARLY data (in-distribution) ────────────────────
print("\n" + "="*60)
print("PHASE 1: Model performance on EARLY data (in-distribution)")
print("="*60)
y_pred_train = model.predict(X_train)
y_prob_train = model.predict_proba(X_train)[:,1]
auc_train = roc_auc_score(y_train, y_prob_train)
report_train = classification_report(y_train, y_pred_train, output_dict=True)
print(f"AUC-ROC : {auc_train:.4f}")
print(f"Recall  : {report_train['1']['recall']:.4f}")
print(f"F1      : {report_train['1']['f1-score']:.4f}")

# ── Evaluate on LATE data (drifted distribution) ─────────────────
print("\n" + "="*60)
print("PHASE 2: Model performance on LATE data (drifted)")
print("="*60)
y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:,1]
auc_test = roc_auc_score(y_test, y_prob_test)
report_test = classification_report(y_test, y_pred_test, output_dict=True)
print(f"AUC-ROC : {auc_test:.4f}")
print(f"Recall  : {report_test['1']['recall']:.4f}")
print(f"F1      : {report_test['1']['f1-score']:.4f}")

# ── Drift magnitude ──────────────────────────────────────────────
recall_drop = report_train['1']['recall'] - report_test['1']['recall']
auc_drop    = auc_train - auc_test
print(f"\n📉 DRIFT IMPACT:")
print(f"  Recall drop : {recall_drop:+.4f}")
print(f"  AUC drop    : {auc_drop:+.4f}")
print(f"  Drift detected: {'YES ⚠️' if abs(recall_drop) > 0.05 else 'NO ✅'}")

# ── Feature distribution drift ───────────────────────────────────
print("\n" + "="*60)
print("PHASE 3: Feature distribution drift analysis")
print("="*60)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()[:20]

drift_report = []
for col in numeric_cols:
    mean_early = X_train[col].mean()
    mean_late  = X_test[col].mean()
    std_early  = X_train[col].std() + 1e-9
    psi        = abs(mean_early - mean_late) / std_early
    drift_report.append({'feature': col, 'mean_early': mean_early,
                         'mean_late': mean_late, 'drift_score': psi})

drift_df = pd.DataFrame(drift_report).sort_values('drift_score', ascending=False)
print("\nTop 10 drifted features:")
print(drift_df.head(10).to_string(index=False))

# ── Simulate new fraud patterns ──────────────────────────────────
print("\n" + "="*60)
print("PHASE 4: New fraud pattern injection")
print("="*60)
np.random.seed(42)
n_new_fraud = 500
new_fraud = X_test.sample(n_new_fraud, replace=True).copy()

# Inject anomalous patterns unseen during training
new_fraud['TransactionAmt'] = np.random.uniform(9000, 15000, n_new_fraud)  # high-value
new_fraud['C1'] = np.random.uniform(50, 200, n_new_fraud)                   # unusual counts
new_fraud['card1'] = np.random.randint(20000, 25000, n_new_fraud)           # new card range

y_new_fraud     = np.ones(n_new_fraud)
y_pred_new      = model.predict(new_fraud)
y_prob_new      = model.predict_proba(new_fraud)[:,1]
new_recall      = (y_pred_new == 1).mean()
print(f"New fraud pattern recall : {new_recall:.4f}")
print(f"Avg confidence score     : {y_prob_new.mean():.4f}")
print(f"Detected as fraud        : {y_pred_new.sum()} / {n_new_fraud}")
print(f"⚠️  Model {'STRUGGLES' if new_recall < 0.5 else 'handles'} new fraud patterns")

# ── Feature importance shift ─────────────────────────────────────
print("\n" + "="*60)
print("PHASE 5: Feature importance shift")
print("="*60)
try:
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    print("Top 10 features (original model):")
    print(feat_imp.head(10).to_string())
except AttributeError:
    print("Note: hybrid model — using XGBoost component importances")
    try:
        importances = model.estimators_[1].feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        print("Top 10 features:")
        print(feat_imp.head(10).to_string())
    except Exception as e:
        print(f"Could not extract importances: {e}")

# ── Save drift report ────────────────────────────────────────────
results = {
    'early_data':  {'auc': round(auc_train,4), 'recall': round(report_train['1']['recall'],4)},
    'late_data':   {'auc': round(auc_test,4),  'recall': round(report_test['1']['recall'],4)},
    'drift':       {'recall_drop': round(recall_drop,4), 'auc_drop': round(auc_drop,4)},
    'new_patterns':{'recall': round(new_recall,4), 'detected': int(y_pred_new.sum())}
}
with open('/home/hamza/A4/monitoring/drift_report.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ DRIFT SIMULATION COMPLETE")
print(f"   Early AUC: {auc_train:.4f} → Late AUC: {auc_test:.4f}")
print(f"   Recall drop: {recall_drop:+.4f}")
print(f"   New fraud pattern recall: {new_recall:.4f}")
print("✅ Report saved to drift_report.json")
print("="*60)
