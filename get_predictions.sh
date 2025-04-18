# # trial 8
# python predict_hvac.py \
#     --model_filename best_model/lstm_hvac_best_trial_8.pth \
#     --feature_scaler_filename scalers/feature_scaler_trial_8.save \
#     --target_scaler_filename scalers/target_scaler_trial_8.save \
#     --training_run_id 09b30ae419144d8a901d993de8277877 \
#     --test_csv dataset/test_features.csv \
#     --results_dir results \
#     --output_filename quan_submission_09b30ae419144d8a901d993de8277877.csv

# # trial 13
# python predict_hvac.py \
#     --model_filename best_model/lstm_hvac_best_trial_13.pth \
#     --feature_scaler_filename scalers/feature_scaler_trial_13.save \
#     --target_scaler_filename scalers/target_scaler_trial_13.save \
#     --training_run_id 5df9205a59354fd5987c1d2dfcaf559b \
#     --test_csv dataset/test_features.csv \
#     --results_dir results \
#     --output_filename quan_submission_5df9205a59354fd5987c1d2dfcaf559b.csv

python predict_hvac.py \
    --model_filename best_model/lstm_hvac_best_seed_42_seed_123_seed_456_seed_789_seed_1011.pth \
    --feature_scaler_filename scalers/feature_scaler_seed_42_seed_123_seed_456_seed_789_seed_1011.save \
    --target_scaler_filename scalers/target_scaler_seed_42_seed_123_seed_456_seed_789_seed_1011.save \
    --training_run_id 5641180a72304d768e1293cee49912e5\
    --test_csv dataset/test_features.csv \
    --results_dir results \
    --output_filename quan_submission_5641180a72304d768e1293cee49912e5.csv
