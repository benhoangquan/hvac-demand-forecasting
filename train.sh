python train.py \
--num_epochs 100 \
--seq_len 32 \
--hidden_size 128 \
--num_layers 4 \
--dropout 0.3 \
--loss_function Huber \
--lr 0.001 \
--early_stopping_patience 15 \
--lr_scheduler_patience 7 \
--no_time_features \
--use_lr_scheduler

python calculate_rmse.py \
--csv_path results/test_set_eval.csv