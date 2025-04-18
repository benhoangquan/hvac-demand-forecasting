python run_pipeline.py \
    --model_filename hybrid_hvac_best.pth \
    --lr 0.0001 \
    --num_epochs 100 \
    --num_layers 4 \
    --batch_size 64 \
    --hidden_size 64 \
    --tcn_num_channels 64 64 \
    --tcn_kernel_size 3 \
    --use_time_features
    
