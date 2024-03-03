if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d  "./logs/forecasting"]; then 
    mkdir ./logs/forecasting
fi

seq_len=336
model_name=PathFormer

random_seed=2024
data_path=./data/Graph_BladeIcing/Icing_128_20/
model_id_name=PathFormer


for pred_len in 96 #192 336 720
do
    python -u run.py \
        --random_seed $random_seed \
        --is_training 1 \
        --data_path $data_path \
        --model_id $model_id_name$pred_len \
        --model $model_name \
        --train_epochs 100 \
        --batch_size 128 \
        --dropout 0.3
done