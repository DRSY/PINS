model=bert-base-uncased
task_name=RTE
epochs=20
warmup_steps=200
initial_warmup=1
final_warmup=6
bs=16
lr=3e-5
beta2=0.99


TEACHER_TYPE=bert
TEACHER_PATH=textattack/bert-base-uncased-${task_name}

CE_LOSS_WEIGHT=1.0
DISTILL_LOSS_WEIGHT=0.0


for seed in 42 12 100 9 2
do
    for final_threshold in 0.20
    do
        CUDA_VISIBLE_DEVICES=0, python -Wignore -u run_glue.py --pruner_name PINS \
        --initial_threshold 1 --final_threshold $final_threshold \
        --warmup_steps $warmup_steps --initial_warmup $initial_warmup --final_warmup $final_warmup \
        --beta1 0.85 --beta2 $beta2 --deltaT 10 \
        --num_train_epochs $epochs --seed $seed --learning_rate $lr \
        --per_gpu_train_batch_size $bs --per_gpu_eval_batch_size 32 \
        --do_train --do_eval --do_lower_case \
        --model_type bert --model_name_or_path $model \
        --logging_steps 50 --save_steps 30000 \
        --data_dir ./data/${task_name} \
        --output_dir ./${model}/${task_name}_${final_threshold}_distil${DISTILL_LOSS_WEIGHT}_${seed} --overwrite_output_dir \
        --task_name $task_name \
        --evaluate_during_training \
        --gradient_accumulation_steps 1 \
        --teacher_name_or_path ${TEACHER_PATH} \
        --teacher_type ${TEACHER_TYPE} \
        --ce_loss_weight ${CE_LOSS_WEIGHT} \
        --distill_loss_weight ${DISTILL_LOSS_WEIGHT}
    done
done