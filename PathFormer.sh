if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d  "./logs/icedetect"]; then 
    mkdir ./logs/icedetect
fi

seq_len=128
model_name=PathFormer
data_name=Icing_128_20/
random_seed=2024
data_path=./data/Graph_BladeIcing/
model_id_name=ice


for pred_len in 2
do
    python -u run.py \
      --is_training 1 \
      --data_path $data_path \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --num_nodes 26 \
      --layer_nums 1 \
      --patch_size_list 16 4 8 32 2 \
      --residual_connection  1\
      --k 2\
      --d_model 8 \
      --d_ff 64 \
      --train_epochs 30\
      --patience 10\
      --lradj 'TST'\
      --itr 1 \
      --batch_size 64 --learning_rate 0.001 >logs/icedetect/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
