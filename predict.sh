python predict_hvac.py \
    --model_filename lstm_hvac_best.pth \
    --use_time_features \
    --hidden_size 64 \
    --num_layers 2 \
    --dropout 0.2 \
    --tcn_num_channels 64 64 \
    --tcn_kernel_size 3 \
    --tcn_dropout 0.2