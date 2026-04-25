import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, precision_score,
                              recall_score, f1_score)
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib, os, warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/home/hamza/A4/ieee-fraud-detection"
MODEL_PATH = "/home/hamza/A4/models"
os.makedirs(MODEL_PATH, exist_ok=True)

# ─────────────────────────────────────────
# LOAD PREPROCESSED DATA
# ─────────────────────────────────────────
print("="*60)
print("Loading preprocessed data")
print("="*60)
df = pd.read_csv(f"{DATA_PATH}/preprocessed.csv")
y  = df["isFraud"]
X  = df.drop(columns=["isFraud"])

print(f"Shape: {X.shape} | Fraud rate: {y.mean()*100:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use best imbalance strategy: Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"After resampling: {X_train_res.shape} | Fraud rate: {y_train_res.mean()*100:.2f}%")

# ─────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────
def evaluate_model(name, model, X_te, y_te, selector=None):
    X_eval = selector.transform(X_te) if selector else X_te
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:,1]

    auc  = roc_auc_score(y_te, y_prob)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"{'='*60}")
    print(f"  AUC-ROC:         {auc:.4f}")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(classification_report(y_te, y_pred, target_names=["Legit","Fraud"]))

    return {
        "Model": name, "AUC-ROC": round(auc,4),
        "Precision": round(prec,4), "Recall": round(rec,4),
        "F1": round(f1,4), "TN": cm[0][0], "FP": cm[0][1],
        "FN": cm[1][0], "TP": cm[1][1]
    }

results = []

# ─────────────────────────────────────────
# MODEL 1: XGBoost
# ─────────────────────────────────────────
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train_res==0).sum()/(y_train_res==1).sum(),
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)
results.append(evaluate_model("XGBoost", xgb, X_test, y_test))
joblib.dump(xgb, f"{MODEL_PATH}/xgboost_model.pkl")

# ─────────────────────────────────────────
# MODEL 2: LightGBM
# ─────────────────────────────────────────
print("\nTraining LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_res, y_train_res)
results.append(evaluate_model("LightGBM", lgbm, X_test, y_test))
joblib.dump(lgbm, f"{MODEL_PATH}/lightgbm_model.pkl")

# ─────────────────────────────────────────
# MODEL 3: Hybrid (RF feature selection + XGBoost)
# ─────────────────────────────────────────
print("\nTraining Hybrid (RF Feature Selection + XGBoost)...")
rf_selector = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1,
    class_weight="balanced", max_depth=8
)
rf_selector.fit(X_train_res, y_train_res)
selector = SelectFromModel(rf_selector, prefit=True, threshold="median")

X_train_sel = selector.transform(X_train_res)
X_test_sel  = selector.transform(X_test)
n_features_selected = X_train_sel.shape[1]
print(f"  Features selected: {n_features_selected} / {X_train_res.shape[1]}")

hybrid_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
hybrid_xgb.fit(X_train_sel, y_train_res)
results.append(evaluate_model("Hybrid (RF+XGBoost)", hybrid_xgb, X_test_sel, y_test))
joblib.dump(hybrid_xgb, f"{MODEL_PATH}/hybrid_model.pkl")
joblib.dump(selector,   f"{MODEL_PATH}/feature_selector.pkl")

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
summary = pd.DataFrame(results)
print(summary[["Model","AUC-ROC","Precision","Recall","F1"]].to_string(index=False))

# Pick best by AUC
best = summary.loc[summary["AUC-ROC"].idxmax()]
print(f"\n🏆 Best Model: {best['Model']} with AUC={best['AUC-ROC']}")

summary.to_csv(f"{DATA_PATH}/model_comparison.csv", index=False)
print(f"✅ Model comparison saved → {DATA_PATH}/model_comparison.csv")
print(f"✅ Models saved → {MODEL_PATH}/")
