import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, precision_score,
                              recall_score, f1_score)
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import joblib, warnings
warnings.filterwarnings("ignore")

DATA_PATH  = "/home/hamza/A4/ieee-fraud-detection"
MODEL_PATH = "/home/hamza/A4/models"

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
print("="*60)
print("TASK 4: Cost-Sensitive Learning")
print("="*60)

df = pd.read_csv(f"{DATA_PATH}/preprocessed.csv")
y  = df["isFraud"]
X  = df.drop(columns=["isFraud"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# ─────────────────────────────────────────
# BUSINESS COST DEFINITION
# ─────────────────────────────────────────
# False Negative (missed fraud) = $500 average fraud loss
# False Positive (false alarm)  = $10 investigation cost
COST_FN = 500
COST_FP = 10

def business_cost(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1][0]
    fp = cm[0][1]
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    return total_cost, fn, fp

def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]
    auc    = roc_auc_score(y_te, y_prob)
    prec   = precision_score(y_te, y_pred)
    rec    = recall_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred)
    cost, fn, fp = business_cost(y_te, y_pred)
    cm     = confusion_matrix(y_te, y_pred)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  AUC-ROC:          {auc:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  💰 Business Impact:")
    print(f"    False Negatives (missed fraud): {fn} × ${COST_FN} = ${fn*COST_FN:,}")
    print(f"    False Positives (false alarms): {fp} × ${COST_FP}  = ${fp*COST_FP:,}")
    print(f"    Total Business Cost:            ${cost:,}")

    return {
        "Model": name, "AUC-ROC": round(auc,4),
        "Precision": round(prec,4), "Recall": round(rec,4),
        "F1": round(f1,4), "FN": fn, "FP": fp,
        "Fraud Loss ($)": fn*COST_FN,
        "False Alarm Cost ($)": fp*COST_FP,
        "Total Cost ($)": cost
    }

results = []

# ─────────────────────────────────────────
# MODEL 1: Standard XGBoost (no cost penalty)
# ─────────────────────────────────────────
print("\nTraining Standard XGBoost...")
xgb_std = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1
)
xgb_std.fit(X_train_res, y_train_res)
results.append(evaluate("Standard XGBoost", xgb_std, X_test, y_test))

# ─────────────────────────────────────────
# MODEL 2: Cost-Sensitive XGBoost
# Higher scale_pos_weight = higher penalty for missing fraud
# ─────────────────────────────────────────
print("\nTraining Cost-Sensitive XGBoost...")
# Cost ratio: FN cost / FP cost = 500/10 = 50
cost_ratio = COST_FN / COST_FP
xgb_cost = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=cost_ratio,
    eval_metric="auc", random_state=42, n_jobs=-1
)
xgb_cost.fit(X_train_res, y_train_res)
results.append(evaluate("Cost-Sensitive XGBoost (scale_pos_weight=50)", xgb_cost, X_test, y_test))

# ─────────────────────────────────────────
# MODEL 3: Cost-Sensitive with custom threshold
# ─────────────────────────────────────────
print("\nCost-Sensitive XGBoost with Optimal Threshold...")
y_prob = xgb_cost.predict_proba(X_test)[:,1]

# Find threshold that minimizes business cost
best_threshold = 0.5
best_cost = float("inf")
threshold_results = []

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_t = (y_prob >= threshold).astype(int)
    cost, fn, fp = business_cost(y_test, y_pred_t)
    threshold_results.append({"threshold": round(threshold,2), "cost": cost, "fn": fn, "fp": fp})
    if cost < best_cost:
        best_cost = cost
        best_threshold = threshold

print(f"  Optimal threshold: {best_threshold:.2f} | Min cost: ${best_cost:,}")
y_pred_optimal = (y_prob >= best_threshold).astype(int)
auc  = roc_auc_score(y_test, y_prob)
prec = precision_score(y_test, y_pred_optimal)
rec  = recall_score(y_test, y_pred_optimal)
f1   = f1_score(y_test, y_pred_optimal)
cost, fn, fp = business_cost(y_test, y_pred_optimal)
cm   = confusion_matrix(y_test, y_pred_optimal)

print(f"\n{'='*60}")
print(f"  Cost-Sensitive XGBoost (Optimal Threshold={best_threshold:.2f})")
print(f"{'='*60}")
print(f"  AUC-ROC:   {auc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  Confusion Matrix:")
print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
print(f"\n  💰 Business Impact:")
print(f"    False Negatives: {fn} × ${COST_FN} = ${fn*COST_FN:,}")
print(f"    False Positives: {fp} × ${COST_FP}  = ${fp*COST_FP:,}")
print(f"    Total Cost:      ${cost:,}")

results.append({
    "Model": f"Cost-Sensitive XGBoost (threshold={best_threshold:.2f})",
    "AUC-ROC": round(auc,4), "Precision": round(prec,4),
    "Recall": round(rec,4), "F1": round(f1,4),
    "FN": fn, "FP": fp,
    "Fraud Loss ($)": fn*COST_FN,
    "False Alarm Cost ($)": fp*COST_FP,
    "Total Cost ($)": cost
})

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*60)
print("COST-SENSITIVE LEARNING SUMMARY")
print("="*60)
summary = pd.DataFrame(results)
print(summary[["Model","AUC-ROC","Recall","F1","FN","FP","Total Cost ($)"]].to_string(index=False))

best = summary.loc[summary["Total Cost ($)"].idxmin()]
print(f"\n🏆 Lowest Business Cost: {best['Model']} at ${best['Total Cost ($)']:,}")

summary.to_csv(f"{DATA_PATH}/cost_sensitive_comparison.csv", index=False)
joblib.dump(xgb_cost, f"{MODEL_PATH}/cost_sensitive_model.pkl")
print(f"\n✅ Cost-sensitive comparison saved")
print(f"✅ Cost-sensitive model saved → {MODEL_PATH}/cost_sensitive_model.pkl")
