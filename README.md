# Fraud Guard MLOps

Fraud review has a capacity cost. A useful score needs a threshold tied to that cost and a way to notice when transaction patterns change.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![CI](https://github.com/nleins5/fraud-guard-mlops/actions/workflows/pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-0f766e)

I generated a chronological transaction stream, trained a model on the early period, and chose a review threshold on a later validation window. The final window changes amount and device-risk patterns to exercise a simple drift monitor.

![Fraud monitoring](artifacts/fraud_monitoring.png)

## Results on simulated data

On the shifted final window, the policy sends 16.04% of transactions to review and catches 65.64% of simulated fraud cases. The threshold assumes a $240 missed-fraud cost and a $20 review cost. Those are scenario inputs, not measured losses from a bank.

| Production result | Value |
|---|---:|
| Average precision | 0.3170 |
| Fraud recall | 65.64% |
| Review rate | 16.04% |
| Selected threshold | 0.0597 |
| Drift alerts | 1 |

Device risk triggers the drift alert. The alert says the input distribution changed; it does not establish why or prove that model performance declined.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
make install
make run
```

The generated stream keeps the experiment repeatable without publishing financial records.

## What I would improve next

I would calibrate probabilities, account for delayed fraud labels, and check whether the review policy works consistently across customer groups.

