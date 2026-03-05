# the evaluation setting
num_processes=2  # the number of gpus you have, e.g., 2
eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/eval.py  # the training configuration
model=ckps/FedSDA/best.pt

# n_sampless=(47147 27490 67847 103016 117068)
n_sampless=(100 100 100 100 100)
cls_numbers=(0 1 2 3 4)

for i in "${!n_sampless[@]}"; do
    cls_number="${cls_numbers[i]}"
    n_samples="${n_sampless[i]}"
    output="logs/FedSDA_${i}.log"
    sample_path="samples/FedSDA_${i}" # path to save sample
    echo "cls_number: $cls_number, n_samples: $n_samples, output: $output, sample_path: $sample_path"
    # accelerate launch --multi_gpu --num_processes $num_processes $eval_script \
    accelerate launch $eval_script \
    --config=$config \
    --nnet_path=$model \
    --output_path=$output \
    --n_samples=$n_samples \
    --sample_path=$sample_path \
    --dataset_path=$dataset_path \
    --cls_number=$cls_number
done
