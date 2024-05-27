#!/bin/bash

# declare -a datasets=("glue-sst2" "glue-rte" "glue-qqp" "glue-wnli")
declare -a datasets=("subj")
declare -a variants=("gold_limit1000" "25_correct" "50_correct" "75_correct" "random" "permutated_labels" "permutated_train_labels")
# declare -a variants=("gold_limit1000")
for variant in "${variants[@]}"; do
    for dataset in "${datasets[@]}"; do
        python create_data.py --variant "$variant" --dataset "$dataset"
    done
done


