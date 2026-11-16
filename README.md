# HVAC demand forecasting

Forecasts a Montreal house's HVAC power draw 4 hours ahead (16 steps of 15 minutes) from outdoor weather, indoor temperature and recent consumption. Course project for IFT 3395 / 6390 (Université de Montréal), scored on a Kaggle competition. Team project: Jae (`dotjae`) added the analysis notebooks.

<!-- demo: add docs/demo.gif and an image link here once the recording exists -->

## Architecture

```
15-min series: outdoor temp, humidity, solar, wind, indoor temp, HVAC kW
      │
      ├─ time features (hour, day, month) + scaling
      ├─ sliding windows of 16 steps
      ▼
LSTM (2 layers, batch norm, dropout) ─▶ HVAC kW 4 h ahead
      ▲
      └─ ensemble: 5 seeds (42, 123, 456, 789, 1011), predictions averaged
```

| File | Role |
|---|---|
| `core_components.py`, `tcn_model.py`, `hybrid_model.py` | Dataset, LSTM, temporal convolutional network, and an LSTM+TCN blend with a learned weight |
| `train.py` | Training loop written from scratch (course rule: no `model.fit()`): Huber or MSE loss, ReduceLROnPlateau, early stopping, gradient clipping, MLflow logging |
| `ensemble_train.py` | Trains one model per seed and records the model paths |
| `optimize.py`, `run_pipeline.py` | Optuna hyperparameter search, and a train-then-predict pipeline |
| `predict_hvac.py` | Loads the saved models and scalers, writes the Kaggle submission |
| `split_dataset.py` | Chronological 80/20 split of the training year |
| `docs/assignment.md` | The course brief |

The final ensemble in `ensemble_train.sh` uses the LSTM alone (`--no_tcn`): 2 layers, 32 hidden units, batch size 32, MSE loss. The hybrid LSTM+TCN is the `train.py` default and was tried during tuning.

## Results

Held-out split: the last 20% of the year (2022-10-20 to 2022-12-31), 6,977 windows. Errors are in kW, computed from the evaluation CSVs in `results/bagging/`.

| Model | RMSE | MAE |
|---|---|---|
| Predict the training mean | 2.15 | 1.46 |
| Persistence (future = current draw) | 0.86 | 0.59 |
| Single LSTM, per seed (5 seeds) | 0.61 to 0.67 | 0.44 to 0.48 |
| 5-seed LSTM ensemble | 0.59 | 0.43 |

Baselines are computed on the same split from `dataset/test_split.csv`. This is a local held-out score, not the Kaggle leaderboard score.

## Quickstart

```bash
pip install -r requirements.txt
python split_dataset.py          # writes dataset/train_split.csv and test_split.csv
bash ensemble_train.sh           # trains the 5-seed ensemble, logs runs to MLflow
python predict_hvac.py --help    # see get_predictions.sh for a full example
```

Place the competition files `train_dataset.csv` and `test_features.csv` in `dataset/` first. `mlflow ui` shows the tracked runs.
