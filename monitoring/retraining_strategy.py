import pandas as pd, numpy as np, joblib, json, time
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings; warnings.filterwarnings('ignore')

print("="*60)
print("TASK 8: INTELLIGENT RETRAINING STRATEGY")
print("="*60)

# ── Load data & model ─────────────────────────────────────────────
df    = pd.read_csv('/home/hamza/A4/ieee-fraud-detection/preprocessed.csv')
model = joblib.load('/home/hamza/A4/models/hybrid_model.pkl')

TARGET    = 'isFraud'
DROP_COLS = [TARGET, 'TransactionDT']
feat_cols = [c for c in df.columns if c not in DROP_COLS]
n_feats   = model.n_features_in_
sel_cols  = feat_cols[:n_feats]

# Time-based splits
dt_33 = df['TransactionDT'].quantile(0.33)
dt_66 = df['TransactionDT'].quantile(0.66)
train_df = df[df['TransactionDT'] <= dt_33].copy()
mid_df   = df[(df['TransactionDT'] > dt_33) & (df['TransactionDT'] <= dt_66)].copy()
test_df  = df[df['TransactionDT'] > dt_66].copy()

X_train = train_df[sel_cols].fillna(0)
y_train = train_df[TARGET]
X_mid   = mid_df[sel_cols].fillna(0)
y_mid   = mid_df[TARGET]
X_test  = test_df[sel_cols].fillna(0)
y_test  = test_df[TARGET]

print(f"Train: {len(X_train):,} | Mid: {len(X_mid):,} | Test: {len(X_test):,}")

def evaluate(model, X, y, threshold=0.45):
    y_prob = model.predict_proba(X)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    return {'auc': round(auc,4),
            'recall': round(rep['1']['recall'],4),
            'precision': round(rep['1']['precision'],4),
            'f1': round(rep['1']['f1-score'],4)}

def retrain_model(X, y):
    m = XGBClassifier(n_estimators=100, max_depth=6,
                      scale_pos_weight=10, random_state=42,
                      eval_metric='logloss', verbosity=0)
    m.fit(X, y)
    return m

# ── Baseline: no retraining ───────────────────────────────────────
print("\n--- STRATEGY 0: No Retraining (Baseline) ---")
t0 = time.time()
base_metrics = evaluate(model, X_test, y_test)
base_time    = time.time() - t0
print(f"AUC: {base_metrics['auc']} | Recall: {base_metrics['recall']} | F1: {base_metrics['f1']}")
print(f"Compute time: {base_time:.2f}s")

# ── Strategy 1: Periodic Retraining ──────────────────────────────
print("\n--- STRATEGY 1: Periodic Retraining ---")
print("Retrains on schedule regardless of performance")
t1 = time.time()
# Simulate: retrain on train+mid data (new period arrived)
X_periodic = pd.concat([X_train, X_mid])
y_periodic = pd.concat([y_train, y_mid])
model_periodic = retrain_model(X_periodic, y_periodic)
periodic_train_time = time.time() - t1
periodic_metrics = evaluate(model_periodic, X_test, y_test)
print(f"AUC: {periodic_metrics['auc']} | Recall: {periodic_metrics['recall']} | F1: {periodic_metrics['f1']}")
print(f"Retrain time: {periodic_train_time:.2f}s")

# ── Strategy 2: Threshold-Based Retraining ────────────────────────
print("\n--- STRATEGY 2: Threshold-Based Retraining ---")
print("Retrains only when recall drops below 0.65")
RECALL_THRESHOLD = 0.65
t2 = time.time()

# Check recall on mid data (monitoring window)
mid_metrics = evaluate(model, X_mid, y_mid)
print(f"Monitored recall on mid data: {mid_metrics['recall']}")

if mid_metrics['recall'] < RECALL_THRESHOLD:
    print(f"Recall {mid_metrics['recall']} < {RECALL_THRESHOLD} — TRIGGERING RETRAIN")
    X_thresh = pd.concat([X_train, X_mid])
    y_thresh = pd.concat([y_train, y_mid])
    model_threshold = retrain_model(X_thresh, y_thresh)
    triggered = True
else:
    print("Recall OK — no retraining needed")
    model_threshold = model
    triggered = False

thresh_train_time = time.time() - t2
thresh_metrics = evaluate(model_threshold, X_test, y_test)
print(f"AUC: {thresh_metrics['auc']} | Recall: {thresh_metrics['recall']} | F1: {thresh_metrics['f1']}")
print(f"Retrain triggered: {triggered} | Time: {thresh_train_time:.2f}s")

# ── Strategy 3: Hybrid (Threshold + Periodic) ────────────────────
print("\n--- STRATEGY 3: Hybrid Strategy (Threshold + Periodic) ---")
print("Retrains if recall drops OR every N periods")
PERIOD_LIMIT = 2  # retrain every 2 periods max
t3 = time.time()

periods_since_retrain = 2  # simulate we haven't retrained in 2 periods
should_retrain = (mid_metrics['recall'] < RECALL_THRESHOLD) or \
                 (periods_since_retrain >= PERIOD_LIMIT)

print(f"Recall trigger: {mid_metrics['recall'] < RECALL_THRESHOLD}")
print(f"Period trigger: {periods_since_retrain >= PERIOD_LIMIT}")
print(f"Retraining: {should_retrain}")

X_hybrid = pd.concat([X_train, X_mid])
y_hybrid = pd.concat([y_train, y_mid])
model_hybrid = retrain_model(X_hybrid, y_hybrid)
hybrid_train_time = time.time() - t3
hybrid_metrics = evaluate(model_hybrid, X_test, y_test)
print(f"AUC: {hybrid_metrics['auc']} | Recall: {hybrid_metrics['recall']} | F1: {hybrid_metrics['f1']}")
print(f"Retrain time: {hybrid_train_time:.2f}s")

# ── Comparison Table ─────────────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY COMPARISON")
print("="*60)
print(f"{'Strategy':<30} {'AUC':>6} {'Recall':>8} {'F1':>6} {'Time(s)':>9} {'Retrains':>9}")
print("-"*70)
print(f"{'0. No Retraining':<30} {base_metrics['auc']:>6} {base_metrics['recall']:>8} {base_metrics['f1']:>6} {base_time:>9.2f} {'0':>9}")
print(f"{'1. Periodic':<30} {periodic_metrics['auc']:>6} {periodic_metrics['recall']:>8} {periodic_metrics['f1']:>6} {periodic_train_time:>9.2f} {'1':>9}")
print(f"{'2. Threshold-Based':<30} {thresh_metrics['auc']:>6} {thresh_metrics['recall']:>8} {thresh_metrics['f1']:>6} {thresh_train_time:>9.2f} {'1' if triggered else '0':>9}")
print(f"{'3. Hybrid':<30} {hybrid_metrics['auc']:>6} {hybrid_metrics['recall']:>8} {hybrid_metrics['f1']:>6} {hybrid_train_time:>9.2f} {'1':>9}")

# ── Business impact analysis ──────────────────────────────────────
print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)
FN_COST = 500   # missed fraud
FP_COST = 10    # false alarm

def business_cost(metrics, y_true, y_prob, threshold=0.45):
    y_pred = (y_prob >= threshold).astype(int)
    fn = ((y_true==1) & (y_pred==0)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    return fn * FN_COST + fp * FP_COST

cost_base     = business_cost(base_metrics,     y_test, model.predict_proba(X_test)[:,1])
cost_periodic = business_cost(periodic_metrics, y_test, model_periodic.predict_proba(X_test)[:,1])
cost_thresh   = business_cost(thresh_metrics,   y_test, model_threshold.predict_proba(X_test)[:,1])
cost_hybrid   = business_cost(hybrid_metrics,   y_test, model_hybrid.predict_proba(X_test)[:,1])

print(f"{'Strategy':<30} {'Cost ($)':>10} {'Savings vs Base':>15}")
print("-"*55)
print(f"{'0. No Retraining':<30} ${cost_base:>9,}")
print(f"{'1. Periodic':<30} ${cost_periodic:>9,} ${cost_base-cost_periodic:>+14,}")
print(f"{'2. Threshold-Based':<30} ${cost_thresh:>9,} ${cost_base-cost_thresh:>+14,}")
print(f"{'3. Hybrid':<30} ${cost_hybrid:>9,} ${cost_base-cost_hybrid:>+14,}")

# ── Save results ──────────────────────────────────────────────────
results = {
    'baseline':   {**base_metrics,     'cost': int(cost_base),     'retrain_time': 0},
    'periodic':   {**periodic_metrics, 'cost': int(cost_periodic), 'retrain_time': round(periodic_train_time,2)},
    'threshold':  {**thresh_metrics,   'cost': int(cost_thresh),   'retrain_time': round(thresh_train_time,2)},
    'hybrid':     {**hybrid_metrics,   'cost': int(cost_hybrid),   'retrain_time': round(hybrid_train_time,2)},
}
with open('/home/hamza/A4/monitoring/retraining_results.json','w') as f:
    json.dump(results, f, indent=2)

joblib.dump(model_hybrid, '/home/hamza/A4/models/retrained_hybrid.pkl')

print("\n✅ TASK 8 COMPLETE")
print("✅ Saved retraining_results.json")
print("✅ Saved retrained_hybrid.pkl")
