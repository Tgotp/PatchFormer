if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d  "./logs/forecasting"]; then 
    mkdir ./logs/forecasting
fi

seq_len = 336
model_name = PathFormer

random_seed = 2024
data_path_name = ./data/Graph_BladeIcing/Icing_128_20/
model_id_name = PathFormer


for pred_len in 96 192 336 720
do
    python -u run.py \
        --random_seed $random_seed \
        --is_traning 1 \
        --data_path_name $data_path_name \
        --model_id $model_id \
        --train_epochs 100 \
        --dropout 0.3
done