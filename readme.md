IFT 3395 - 6390 Term Project
The term project takes place on the Kaggle platform. You can join the competition at the
following link:
https://www.kaggle.com/t/36154c3788c24a23b77b7b59a93a7b58
Detailed instructions are available directly on Kaggle.

Context
In Quebec, energy consumption typically peaks daily in the morning, between 6 a.m. and 9 a.m., and in the evening, between 4 p.m. and 7 p.m. These periods, called peak periods, are particularly challenging to manage, especially during extreme weather conditions, whether very hot or very cold. To reduce these peaks, Hydro-Québec offers advantageous rates to encourage users to limit their consumption during these critical periods.

An effective method to reduce consumption during peak periods is to adjust heating or air conditioning, which represent the main sources of energy consumption in a building. Thanks to thermal inertia, it is possible to use energy before the peak period to preheat or precool the building while preserving occupant comfort. To optimize these strategies, it is necessary to predict, a few hours in advance, the consumption of the heating, ventilation, and air conditioning (HVAC) system. These predictions allow anticipating peaks and implementing proactive heating control rather than reactive control.

In this project, you must design a predictive model for energy consumption in a house located in
Montreal. You have access to one year of data containing the following features:
● Date
● HVAC system consumption (kW)
● Indoor temperature (°C)
● Weather data:
○ Outdoor temperature (°C)
○ Humidity (%)
○ Solar radiation (W/m²)
○ Wind speed (km/h)
○ Wind direction (°)
Using the method of your choice, develop a model to predict HVAC system consumption up to 4
hours (16 time steps) in advance.
Allowed Libraries
You may use any automatic differentiation library (TensorFlow, JAX, PyTorch, etc.) to facilitate
the implementation of your method. However, you must implement the training loops yourself
(you should not use pre-built models and simply call model.fit()).
Note:
Submission Requirements
In addition to your submission on Kaggle, you must submit a Google Colab notebook
containing detailed explanations of your methodology, the code used, and graphs demonstrating
that your method works correctly. Please submit a PDF on Gradescope that contains the link to
your Google Colab, and make sure it is visible to us. Please add your teammates on
Gradescope as well, just as you did for HW1.

## First LSTM Prototype

A simple LSTM-based prototype script is provided in `hvac_lstm_prototype.py` for predicting HVAC consumption 4 hours (16 time steps) ahead.

To run the prototype:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Execute the script:

   ```bash
   python hvac_lstm_prototype.py
   ```

The script will train a baseline LSTM model and save the trained model and feature scaler in the `models/` directory.
  
## Prediction on Test Set
  
Once training is complete, you can generate predictions on the test set:

```bash
python predict_hvac.py
```

This will read `dataset/test_features.csv`, use the trained LSTM to predict HVAC consumption for each `ID`,
and write the results to `results/test1.csv` with columns `ID,prediction`.

## Splitting Dataset 80/20

To create a local train/test split (80% training, 20% test) based on chronological order:

```bash
python split_dataset.py
```

This script reads `dataset/train_dataset.csv`, sorts by `date`, and writes:
- `dataset/train_split.csv` (first 80% of rows)
- `dataset/test_split.csv` (last 20% of rows)

## Train & Evaluate on the Split

With the splits created, you can train an LSTM on the 80% train set and evaluate on the 20% test set:

```bash
python evaluate_hvac_split.py --num_epochs 10 --batch_size 64
```

By default, it uses `dataset/train_split.csv` and `dataset/test_split.csv`, and will:
- Print epoch‐wise training MSE
- Print final Test MSE and MAE
- Save per‐sample predictions vs. ground truth to `results/train_test_eval.csv`