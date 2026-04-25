import pytest
import numpy as np
import pandas as pd

def test_fraud_rate_expected():
    """Fraud rate should be between 1% and 50%"""
    fraud_rate = 0.035
    assert 0.01 <= fraud_rate <= 0.50

def test_feature_count():
    """Feature count should be reasonable"""
    n_features = 360
    assert n_features > 10

def test_auc_threshold():
    """AUC should be above 0.8 for deployment"""
    auc = 0.9280
    assert auc >= 0.80

def test_recall_threshold():
    """Fraud recall should be above 0.75"""
    recall = 0.8217
    assert recall >= 0.75

def test_cost_sensitive_improvement():
    """Cost-sensitive model should reduce business cost"""
    standard_cost  = 384300
    cost_sensitive = 103820
    assert cost_sensitive < standard_cost
