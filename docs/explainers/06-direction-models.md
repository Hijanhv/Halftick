# Direction models (`models/`)

## What they do

Estimate P(the next mid move is up) at any event, as a calibrated probability.

| Model | What it is | Why it is here |
|---|---|---|
| Baseline A | Split queue imbalance into ten deciles; the prediction is the historical share of up moves in that decile (with a small Laplace correction) | The Gould & Bonart benchmark. No parameters to overfit. If nothing beats it, that is a finding. |
| Baseline B | Logistic regression on imbalance, microprice and OFI (50 events) | The simplest model that combines the three classic signals |
| LightGBM | Gradient-boosted trees on all features | Can pick up non-linear effects and interactions (for example, imbalance matters more when the queues are small) |

## Calibration

The simulator plugs these probabilities into cost comparisons, so a "70%" has to mean 70%. Each model is calibrated on a separate held-out day with **isotonic regression**, a monotone step function fitted from raw score to observed frequency. Platt scaling (a logistic fit on the logit of the score) is available as an option. Reliability diagrams show the result.

## Walk-forward validation

Events a few milliseconds apart are almost identical, so a random train/test split would let the test set "see" its neighbours in training and give flattering scores. Instead:

```
train on days 1..k  ->  calibrate on day k+1  ->  test on day k+2   (then k = k+1)
```

Nothing is ever shuffled across days, the training window only grows forward, and a property-based test checks across random configurations that no fold ever trains or calibrates on a day at or after its test day.

Each day contributes a random sample of 150,000 rows (configurable). That keeps training fast and stops busy days from dominating.

## Metrics

Log loss and Brier score (are the probabilities right?), AUC (does the model rank up-moves above down-moves?), and accuracy at 0.5 (easiest to explain). In-sample numbers are reported next to out-of-sample ones so overfitting is visible.

## Is LightGBM really better?

Small differences in log loss can be noise, and the rows are autocorrelated, so a naive t-test would overstate significance. The per-row log-loss difference between two models is tested with a **Newey-West (HAC)** standard error, which allows for that autocorrelation.

## What could go wrong

* The hidden signal in the synthetic data was built to be learnable from order flow, so LightGBM's edge over the baselines here says more about the generator than about ZN.
* Calibration on a single day is noisy. Isotonic regression can overfit a small calibration set; Platt is the fallback.

## Alternatives considered

* **sklearn's CalibratedClassifierCV**: its prefit mode is being deprecated, and fitting IsotonicRegression directly is clearer.
* **Deep learning (DeepLOB-style CNNs/LSTMs)**: ruled out by the spec. For tabular features and this data size, gradient boosting is the standard strong choice.
