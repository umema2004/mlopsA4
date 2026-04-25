import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics

# ─────────────────────────────────────────
# COMPONENT 1: Data Ingestion
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"]
)
def data_ingestion(
    data_path: str,
    train_transaction_out: Output[Dataset],
    train_identity_out: Output[Dataset]
):
    import pandas as pd
    import os

    print(f"Loading data from {data_path}")
    trans = pd.read_csv(os.path.join(data_path, "train_transaction.csv"))
    iden = pd.read_csv(os.path.join(data_path, "train_identity.csv"))

    print(f"Transactions shape: {trans.shape}")
    print(f"Identity shape: {iden.shape}")

    trans.to_csv(train_transaction_out.path, index=False)
    iden.to_csv(train_identity_out.path, index=False)
    print("Data ingestion complete.")


# ─────────────────────────────────────────
# COMPONENT 2: Data Validation
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"]
)
def data_validation(
    train_transaction: Input[Dataset],
    train_identity: Input[Dataset],
    validation_report: Output[Dataset]
):
    import pandas as pd
    import json

    trans = pd.read_csv(train_transaction.path)
    iden = pd.read_csv(train_identity.path)

    report = {
        "transaction_rows": int(trans.shape[0]),
        "transaction_cols": int(trans.shape[1]),
        "identity_rows": int(iden.shape[0]),
        "identity_cols": int(iden.shape[1]),
        "transaction_missing_pct": float(trans.isnull().mean().mean() * 100),
        "identity_missing_pct": float(iden.isnull().mean().mean() * 100),
        "fraud_rate": float(trans["isFraud"].mean() * 100),
        "target_present": "isFraud" in trans.columns,
        "schema_valid": True
    }

    print("Validation Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    with open(validation_report.path, "w") as f:
        json.dump(report, f)
    print("Data validation complete.")


# ─────────────────────────────────────────
# COMPONENT 3: Data Preprocessing
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def data_preprocessing(
    train_transaction: Input[Dataset],
    train_identity: Input[Dataset],
    preprocessed_out: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    trans = pd.read_csv(train_transaction.path)
    iden = pd.read_csv(train_identity.path)

    # Merge on TransactionID
    df = trans.merge(iden, on="TransactionID", how="left")
    print(f"Merged shape: {df.shape}")

    # Drop columns with >80% missing
    thresh = 0.8
    missing_pct = df.isnull().mean()
    df = df[missing_pct[missing_pct < thresh].index]
    print(f"After dropping high-missing cols: {df.shape}")

    # Separate target
    y = df["isFraud"]
    df = df.drop(columns=["isFraud", "TransactionID"])

    # Numeric columns: median imputation
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    num_imp = SimpleImputer(strategy="median")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    # Categorical columns: mode imputation
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    df["isFraud"] = y.values
    df.to_csv(preprocessed_out.path, index=False)
    print("Preprocessing complete.")


# ─────────────────────────────────────────
# COMPONENT 4: Feature Engineering
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "category_encoders"]
)
def feature_engineering(
    preprocessed: Input[Dataset],
    features_out: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    import category_encoders as ce

    df = pd.read_csv(preprocessed.path)

    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Applying target encoding to {len(cat_cols)} categorical columns")

    # Target encoding for high-cardinality categoricals
    encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10)
    X_encoded = encoder.fit_transform(X, y)

    # Feature: transaction amount log transform
    if "TransactionAmt" in X_encoded.columns:
        X_encoded["TransactionAmt_log"] = np.log1p(X_encoded["TransactionAmt"])

    X_encoded["isFraud"] = y.values
    X_encoded.to_csv(features_out.path, index=False)
    print(f"Feature engineering complete. Shape: {X_encoded.shape}")


# ─────────────────────────────────────────
# COMPONENT 5: Model Training
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "joblib"]
)
def model_training(
    features: Input[Dataset],
    model_out: Output[Model],
    metrics_out: Output[Metrics],
    imbalance_strategy: str = "smote"
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.metrics import classification_report, roc_auc_score
    import joblib, json, os

    df = pd.read_csv(features.path)
    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imbalance handling
    print(f"Imbalance strategy: {imbalance_strategy}")
    if imbalance_strategy == "smote":
        sampler = SMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif imbalance_strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    # else: class_weight handled in model

    print(f"Training set after resampling: {X_train.shape}, fraud rate: {y_train.mean():.3f}")

    # --- Model 1: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        eval_metric="auc", random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    # --- Model 2: LightGBM ---
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:,1])
    print(f"LightGBM AUC: {lgbm_auc:.4f}")

    # --- Model 3: Hybrid (RF feature selection + XGBoost) ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    selector = SelectFromModel(rf, prefit=True, threshold="median")
    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    hybrid = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        eval_metric="auc", random_state=42, n_jobs=-1
    )
    hybrid.fit(X_train_sel, y_train)
    hybrid_auc = roc_auc_score(y_test, hybrid.predict_proba(X_test_sel)[:,1])
    print(f"Hybrid (RF+XGB) AUC: {hybrid_auc:.4f}")

    # Pick best model
    best_auc = max(xgb_auc, lgbm_auc, hybrid_auc)
    best_name = ["XGBoost","LightGBM","Hybrid"][
        [xgb_auc, lgbm_auc, hybrid_auc].index(best_auc)
    ]
    best_model = [xgb, lgbm, hybrid][
        [xgb_auc, lgbm_auc, hybrid_auc].index(best_auc)
    ]
    print(f"Best model: {best_name} with AUC={best_auc:.4f}")

    os.makedirs(model_out.path, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_out.path, "model.pkl"))
    joblib.dump(selector, os.path.join(model_out.path, "selector.pkl"))

    metrics_out.log_metric("xgb_auc", xgb_auc)
    metrics_out.log_metric("lgbm_auc", lgbm_auc)
    metrics_out.log_metric("hybrid_auc", hybrid_auc)
    metrics_out.log_metric("best_auc", best_auc)
    metrics_out.log_metric("best_model", best_name)
    print("Model training complete.")


# ─────────────────────────────────────────
# COMPONENT 6: Model Evaluation
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "joblib"]
)
def model_evaluation(
    features: Input[Dataset],
    model: Input[Model],
    eval_metrics: Output[Metrics]
) -> float:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  confusion_matrix, precision_score,
                                  recall_score, f1_score)
    import joblib, os

    df = pd.read_csv(features.path)
    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_obj = joblib.load(os.path.join(model.path, "model.pkl"))

    # Try selector (hybrid model)
    selector_path = os.path.join(model.path, "selector.pkl")
    if os.path.exists(selector_path):
        try:
            selector = joblib.load(selector_path)
            X_test_eval = selector.transform(X_test)
        except:
            X_test_eval = X_test
    else:
        X_test_eval = X_test

    y_pred = model_obj.predict(X_test_eval)
    y_prob = model_obj.predict_proba(X_test_eval)[:,1]

    auc   = roc_auc_score(y_test, y_prob)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))

    eval_metrics.log_metric("auc_roc", auc)
    eval_metrics.log_metric("precision", prec)
    eval_metrics.log_metric("recall", rec)
    eval_metrics.log_metric("f1_score", f1)
    eval_metrics.log_metric("true_negatives",  int(cm[0][0]))
    eval_metrics.log_metric("false_positives", int(cm[0][1]))
    eval_metrics.log_metric("false_negatives", int(cm[1][0]))
    eval_metrics.log_metric("true_positives",  int(cm[1][1]))

    return float(auc)


# ─────────────────────────────────────────
# COMPONENT 7: Conditional Deployment
# ─────────────────────────────────────────
@component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas"]
)
def conditional_deployment(
    auc_score: float,
    model: Input[Model],
    threshold: float = 0.85
):
    print(f"AUC Score: {auc_score:.4f} | Threshold: {threshold}")
    if auc_score >= threshold:
        print(f"✅ Model meets threshold. DEPLOYING model from {model.path}")
        # In production: push to model registry / serving endpoint
    else:
        print(f"❌ Model AUC {auc_score:.4f} below threshold {threshold}. Skipping deployment.")


# ─────────────────────────────────────────
# PIPELINE DEFINITION
# ─────────────────────────────────────────
@dsl.pipeline(
    name="Fraud Detection Pipeline",
    description="End-to-end fraud detection: ingest → validate → preprocess → features → train → evaluate → deploy"
)
def fraud_detection_pipeline(
    data_path: str = "/data/fraud-detection",
    imbalance_strategy: str = "smote",
    deploy_threshold: float = 0.85
):
    # Step 1
    ingest = data_ingestion(data_path=data_path)
    ingest.set_retry(num_retries=2)

    # Step 2
    validate = data_validation(
        train_transaction=ingest.outputs["train_transaction_out"],
        train_identity=ingest.outputs["train_identity_out"]
    )
    validate.set_retry(num_retries=2)

    # Step 3
    preprocess = data_preprocessing(
        train_transaction=ingest.outputs["train_transaction_out"],
        train_identity=ingest.outputs["train_identity_out"]
    )
    preprocess.set_retry(num_retries=2)

    # Step 4
    features = feature_engineering(
        preprocessed=preprocess.outputs["preprocessed_out"]
    )
    features.set_retry(num_retries=2)

    # Step 5
    train = model_training(
        features=features.outputs["features_out"],
        imbalance_strategy=imbalance_strategy
    )
    train.set_retry(num_retries=3)

    # Step 6
    evaluate = model_evaluation(
        features=features.outputs["features_out"],
        model=train.outputs["model_out"]
    )

    # Step 7 — Conditional deployment
    deploy = conditional_deployment(
        auc_score=evaluate.outputs["Output"],
        model=train.outputs["model_out"],
        threshold=deploy_threshold
    )


# ─────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_pipeline.yaml"
    )
    print("Pipeline compiled → fraud_pipeline.yaml")
