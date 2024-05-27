log_path="/srv/home/zxu444/browse/rethink_icl/out/llama-2-13b-chat"

# Define your arrays for tasks and suffixes
declare -a tasks=("glue-rte" "glue-sst2" "glue-qqp" "glue-wnli" "subj")  # Add all your tasks here
declare -a suffixes=("gold_limit1000" "permutated_train_labels" "25_correct" "50_correct" "75_correct")  # Add all your suffixes here

# Loop and execute your command
for task in "${tasks[@]}"
do
       python tools/gather_result.py \
        --condition "{'task': '${task}_gold_limit1000', 'num_fewshot': 0}" \
        --log_path "$log_path"

       for suffix in "${suffixes[@]}"
       do
       condition="{'task': '${task}_${suffix}', 'num_fewshot': 16}"
       python tools/gather_result.py \
              --condition "$condition" \
              --log_path "$log_path"
       done
done
