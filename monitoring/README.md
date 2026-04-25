# Task 6: Monitoring & Observability

## Components
- **Prometheus**: Scrapes fraud API metrics every 15s
- **Grafana**: 3 dashboards (System Health, Model Performance, Data Drift)
- **Alert Rules**: 5 rules (FraudRecallDropped, HighFPR, APILatency, ErrorRate, DataDrift)

## Dashboards
1. Fraud Detection - System Health
2. Fraud Detection - Model Performance  
3. Fraud Detection - Data Drift

## Alert Rules
- FraudRecallDropped: fires when recall < 0.65
- HighFalsePositiveRate: fires when FPR > 0.30
- APIHighLatency: fires when latency > 500ms
- APIErrorRateHigh: fires when error rate > 0.1 req/s
- DataDriftDetected: fires when avg confidence < 0.20

## Evidence
- FraudRecallDropped alert FIRED at recall=0.33
- fraud-inference-api target: UP
- 500+ predictions processed
