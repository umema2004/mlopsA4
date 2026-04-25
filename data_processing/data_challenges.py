import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import category_encoders as ce
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/home/hamza/A4/ieee-fraud-detection"

# ─────────────────────────────────────────
# 1. LOAD & MERGE (sampled)
# ─────────────────────────────────────────
print("="*60)
print("STEP 1: Loading Data (sampled for memory efficiency)")
print("="*60)

trans = pd.read_csv(f"{DATA_PATH}/train_transaction.csv")
iden  = pd.read_csv(f"{DATA_PATH}/train_identity.csv")

# Stratified sample: keep all fraud + sample legit
fraud    = trans[trans["isFraud"] == 1]
legit    = trans[trans["isFraud"] == 0].sample(n=min(60000, len(trans[trans["isFraud"]==0])), random_state=42)
trans    = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

df = trans.merge(iden, on="TransactionID", how="left")
print(f"Sampled merged shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# ─────────────────────────────────────────
# 2. MISSING VALUE HANDLING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Missing Value Analysis & Handling")
print("="*60)

missing = df.isnull().mean().sort_values(ascending=False)
print(f"Columns with >80% missing: {(missing > 0.8).sum()}")
print(f"Columns with >50% missing: {(missing > 0.5).sum()}")
print(f"Columns with >20% missing: {(missing > 0.2).sum()}")

# Drop columns with >80% missing
drop_cols = missing[missing > 0.8].index.tolist()
df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} high-missing columns")
print(f"Shape after drop: {df.shape}")

y = df["isFraud"]
df = df.drop(columns=["isFraud", "TransactionID"])

# Median imputation for numeric
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Mode imputation for categorical
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
    df[col] = df[col].fillna(mode_val)

print(f"Remaining missing values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. HIGH-CARDINALITY ENCODING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: High-Cardinality Categorical Encoding")
print("="*60)

high_card = [c for c in cat_cols if df[c].nunique() > 10]
low_card  = [c for c in cat_cols if df[c].nunique() <= 10]

print(f"High-cardinality cols (target encoding): {len(high_card)}")
print(f"Low-cardinality cols (label encoding):   {len(low_card)}")

encoder = ce.TargetEncoder(cols=high_card, smoothing=10)
df = encoder.fit_transform(df, y)

le = LabelEncoder()
for col in low_card:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"Encoding complete. Shape: {df.shape}")

# ─────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Feature Engineering")
print("="*60)

if "TransactionAmt" in df.columns:
    df["TransactionAmt_log"]     = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_rounded"] = (df["TransactionAmt"] % 1 == 0).astype(int)

print(f"Features after engineering: {df.shape[1]}")

# ─────────────────────────────────────────
# 5. CLASS IMBALANCE COMPARISON
# ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Class Imbalance Handling - Strategy Comparison")
print("="*60)

X = df.values.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original train fraud rate: {y_train.mean()*100:.2f}%")

results = []

def evaluate(name, X_tr, y_tr, X_te, y_te, class_weight=None):
    clf = RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1,
        class_weight=class_weight, max_depth=8
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:,1]
    auc    = roc_auc_score(y_te, y_prob)
    report = classification_report(y_te, y_pred,
                                   target_names=["Legit","Fraud"],
                                   output_dict=True)
    print(f"\n--- {name} ---")
    print(f"  Train size:      {X_tr.shape[0]} | Fraud rate: {y_tr.mean()*100:.2f}%")
    print(f"  AUC-ROC:         {auc:.4f}")
    print(f"  Fraud Recall:    {report['Fraud']['recall']:.4f}")
    print(f"  Fraud Precision: {report['Fraud']['precision']:.4f}")
    print(f"  Fraud F1:        {report['Fraud']['f1-score']:.4f}")
    results.append({
        "Strategy":        name,
        "Train Size":      X_tr.shape[0],
        "Fraud Rate":      f"{y_tr.mean()*100:.2f}%",
        "AUC-ROC":         round(auc, 4),
        "Fraud Recall":    round(report["Fraud"]["recall"], 4),
        "Fraud Precision": round(report["Fraud"]["precision"], 4),
        "Fraud F1":        round(report["Fraud"]["f1-score"], 4),
    })

# Strategy 1: Class weight
evaluate("Class Weighting", X_train, y_train, X_test, y_test, class_weight="balanced")

# Strategy 2: SMOTE
X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
evaluate("SMOTE", X_sm, y_sm, X_test, y_test)

# Strategy 3: Random Undersampling
X_ru, y_ru = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
evaluate("Random Undersampling", X_ru, y_ru, X_test, y_test)

# Strategy 4: SMOTETomek
X_st, y_st = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
evaluate("SMOTETomek (Hybrid)", X_st, y_st, X_test, y_test)

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save outputs
df["isFraud"] = y.values
df.to_csv(f"{DATA_PATH}/preprocessed.csv", index=False)
results_df.to_csv(f"{DATA_PATH}/imbalance_comparison.csv", index=False)
print(f"\n✅ Preprocessed data saved → {DATA_PATH}/preprocessed.csv")
print(f"✅ Imbalance comparison  saved → {DATA_PATH}/imbalance_comparison.csv")
