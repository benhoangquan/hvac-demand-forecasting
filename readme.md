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

Okay, let's break down strategies to improve your HVAC prediction accuracy. Your LSTM approach is sensible given the time-series nature and the puissance_cvac_future target.
Here's an evaluation of your ideas and some additional ones, ranked by estimated implementation ease (simplest first):
Evaluation & Ranking:


Extend Lookback Window (seq_len to 96):
Concept: Provide the LSTM with more historical context (96 steps instead of 16).
Pros: Could capture longer-term dependencies or seasonality missed by the shorter window.
Cons: Significantly increases computational load (memory/time per sequence). May require reducing batch size. Could potentially introduce noise if very old data isn't relevant. Risk of vanishing/exploding gradients increases with sequence length, though LSTMs are designed to mitigate this better than simple RNNs.
Ease: Easy. Primarily involves changing the seq_len value in your config dictionary and rerunning the pipeline. You might need to adjust batch_size down if you hit memory limits.


Feature Engineering (Adding Lags/Rolling Stats):
Concept: Explicitly add features like previous puissance_cvac values (e.g., t-1, t-2, t-16) or rolling window statistics (e.g., mean/std of temperature_interieure over the last 6 hours) to the input features before creating sequences.
Pros: Can make short-term dependencies or recent trends more explicit for the model, potentially helping even if the LSTM could learn them implicitly. Can incorporate domain knowledge.
Cons: Increases the dimensionality of the input features. Requires careful implementation in the data preparation stage (load_and_prepare_data). Need to decide which features and lag/window sizes are relevant.
Ease: Moderate. Requires modifying Pandas DataFrame operations in load_and_prepare_data before scaling and sequence creation. Need to update base_feature_cols and ensure correct handling during scaling and prediction.


Experiment with Loss Function (Handling Zeros/Spikes):
Concept: You're using Huber loss, which is a good start as it's less sensitive to outliers than MSE. To specifically address the zero-prediction issue or spike importance:
Try MAE: nn.L1Loss() might inherently handle the zero predictions differently or be less influenced by the distribution shape.
Tune Huber Delta: The delta parameter in nn.HuberLoss controls the transition point between quadratic and linear loss. Experimenting with this value could change the model's sensitivity.
Weighted Loss (Simpler version): Instead of complex horizon weighting, you could try a simple weighting based on the target value magnitude. E.g., calculate Huber loss, then multiply the loss for each sample by (1 + alpha * target_value) before averaging. This requires a custom loop or modification within the loss calculation.
Pros: Directly targets the observed problems (zeros, spikes). MAE/Huber tuning is simple config change.
Cons: Custom weighting requires careful implementation to ensure gradients flow correctly. Finding the right weighting scheme might require experimentation.
Ease: Easy (MAE/Huber Tuning) / Moderate (Simple Weighting). Changing loss_function in config is easy. Implementing simple weighting requires modifying the training loop slightly or creating a simple custom loss wrapper.


Adding Tree-Based Models (XGBoost/LightGBM):
Concept: Train a gradient-boosted tree model alongside or instead of the LSTM.
Pros: XGBoost/LightGBM excel at tabular data, are often robust, and can capture complex interactions the LSTM might miss. They might handle the zero values or spikes differently.
Cons: Standard tree models don't inherently process sequences. You'd need to restructure the input: either flatten the sequences or, more effectively, create time-series features (lags, rolling stats, time features) for each prediction point. This significantly changes the data preparation pipeline compared to the LSTM's sequence input. Ensembling LSTM + XGBoost adds complexity.
Ease: Moderate-to-Complex. Requires installing new libraries (xgboost, lightgbm). Substantial changes needed in load_and_prepare_data and create_sequences (likely removing create_sequences for XGBoost and creating flat features instead). Training loop structure is different.


Horizon-Specific Loss Weighting (Your Idea):
Concept: Give more importance to errors on high target values (puissance_cvac_future). This is a more targeted version of the weighted loss mentioned in point 3.
Pros: Directly incentivizes the model to predict large spikes accurately.
Cons: Requires implementing a custom loss function in PyTorch. You need to define precisely how the weight should scale with the target value. Needs careful testing.
Ease: Complex. Requires defining a new nn.Module for the loss function, incorporating the weighting logic based on the target tensor (yb), and ensuring correct reduction (mean/sum).


Two-Stage Modeling (Zero Handling):
Concept: First, train a classifier to predict Pr(puissance_cvac_future > 0). Second, train a regression model (like your LSTM, or XGBoost) only on the data points where puissance_cvac_future > 0. Final prediction = Pr(>0) * Regression_prediction.
Pros: Explicitly models the 'on'/'off' behaviour, which might be fundamentally different from predicting the magnitude. Could significantly improve zero predictions.
Cons: Doubles the modeling effort. Requires managing two separate models, datasets (the regression model trains on a subset), and combining predictions carefully. Potential for error propagation.
Ease: Complex. Involves significant structural changes: separate data preparation, training loops, and prediction logic for classification and regression components.


Model Architecture Changes (Attention, BiLSTM, etc.):
Concept: Enhance the LSTM itself, e.g., adding Attention layers to focus on relevant time steps, using Bidirectional LSTMs to process sequences forward and backward, or combining with CNNs.
Pros: Can potentially improve the model's ability to learn complex temporal patterns.
Cons: Increases model complexity. Requires modifying the LSTMModel class definition and potentially retraining Optuna as optimal hyperparameters might change. Needs deeper understanding of these architectures.
Ease: Complex. Requires modifying the nn.Module definition, understanding how these layers interact, and potentially significant tuning.


Summary & Recommendations:
Start with changes that leverage your existing pipeline structure:
Try seq_len = 32 or 48: See if more context helps without excessive computational cost. (Easy)
Switch Loss to MAE: Change config['loss_function'] = 'MAE' (need to add nn.L1Loss handling in run_training_trial). See if it handles zeros better. (Easy)
Add Lagged Features: Add puissance_cvac from t-1, t-16 as input features. (Moderate)
If these don't yield sufficient gains, then consider the more involved options:
XGBoost: Provides a different modeling perspective but requires more rework of the data pipeline. (Moderate-to-Complex)
Targeted Loss Weighting / Two-Stage Model / Architecture Changes: These are powerful but require more significant implementation effort and expertise. (Complex)