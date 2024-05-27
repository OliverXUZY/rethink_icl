import os
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--out_dir", type=str, required=True)
# parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--num_gpus", type=int, default=1)

args = parser.parse_args()

datasets = [#"glue-mrpc", 
            "glue-rte", 
            #"tweet_eval-hate", 
            "glue-sst2",
            "glue-qqp",
            'subj',
            #"openbookqa", "commonsense_qa", "superglue-copa"
            ]
model_list = [ #('gpt2-large', 'out/gpt2-large'),
              #('meta-llama/Llama-2-13b-chat-hf', 'out/llama-2-13b-chat'),
              #('meta-llama/Llama-2-7b-chat-hf', 'out/llama-2-7b-chat'),
              #('meta-llama/Llama-2-70b-chat-hf', 'out/llama-2-70b-chat'),
              ('openlm-research/open_llama_3b_v2', 'out/open_llama-2-3b'),
              #('huggyllama/llama-30b', 'out/llama-30b'), 
              #('huggyllama/llama-7b', 'out/llama-7b'),
              #('huggyllama/llama-13b', 'out/llama-13b'), 
              #('huggyllama/llama-65b', 'out/llama-65b')
              ]

total_num = len(model_list)*len(datasets)*4
for i, dataset_name in enumerate(datasets):
    for j, (base_model_path, out_dir) in enumerate(model_list):
        command = f"python llm_test.py --dataset {dataset_name} --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus}"
        print(f"{(i*len(model_list)+j)*4} / {total_num}")
        print(command)
        os.system(command)
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name} --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*4+1} / {total_num}")
        print(command)
        os.system(command)
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_random --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*4+2} / {total_num}")
        print(command)
        os.system(command)
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_permutated_labels --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*4+3} / {total_num}")
        print(command)
        os.system(command)
        print("\n")
