import os
import argparse
import time
parser = argparse.ArgumentParser()
# parser.add_argument("--out_dir", type=str, required=True)
# parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--num_gpus", type=int, default=1)

args = parser.parse_args()
def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600) # time in hours
  if t >= 60:
    return '{:.1f}m'.format(t / 60)   # time in minutes
  return '{:.1f}s'.format(t)
datasets = [#"glue-mrpc", 
            "glue-rte", 
            #"tweet_eval-hate", 
            "glue-sst2",
            "glue-qqp",
            "glue-wnli",
            'subj',
            #"openbookqa", "commonsense_qa", "superglue-copa"
            ]
model_list = [ #('gpt2-large', 'out/gpt2-large'),
              ('meta-llama/Llama-2-13b-chat-hf', 'out/llama-2-13b-chat'),
            # ('openlm-research/open_llama_3b_v2', 'out/open_llama-2-3b'),
            # ('meta-llama/Llama-2-7b-chat-hf', 'out/llama-2-7b-chat'),
            #   ('meta-llama/Llama-2-70b-chat-hf', 'out/llama-2-70b-chat'),
              #('huggyllama/llama-30b', 'out/llama-30b'), 
              #('huggyllama/llama-7b', 'out/llama-7b'),
              #('huggyllama/llama-13b', 'out/llama-13b'), 
              #('huggyllama/llama-65b', 'out/llama-65b')
              ]

total_num = len(model_list)*len(datasets)*6
count = 0
start = time.time()
for i, dataset_name in enumerate(datasets):
    for j, (base_model_path, out_dir) in enumerate(model_list):
        command = f"python llm_test.py --dataset {dataset_name}_gold_limit1000 --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus}"
        print(f"{(i*len(model_list)+j)*6} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_gold_limit1000 --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*6+1} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_permutated_train_labels --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*6+2} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))        
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_25_correct --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*6+3} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))        
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_50_correct --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*6+4} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))        
        print("\n")

        command = f"python llm_test.py --dataset {dataset_name}_75_correct --base_model_path {base_model_path} --method direct --out_dir {out_dir} --do_zeroshot --test_batch_size 8 --num_gpus {args.num_gpus} --use_demonstrations --k 16 --seed 100,13,21,42,87"
        print(f"{(i*len(model_list)+j)*6+5} / {total_num}")
        print(command)
        os.system(command)
        count += 1
        print("time elapse {} | {}".format(time_str(time.time() - start), time_str((time.time() - start)/count*total_num)))        
        print("\n")
