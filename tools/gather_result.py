import sys
sys.path.append("/srv/home/zxu444/browse/rethink_icl")
import argparse
import os
import numpy as np

def parse_str_to_dict(d):
    dictionary = dict()
    # Removes curly braces and splits the pairs into a list
    pairs = d.strip('\n').strip('{}').split(', ')
    for i in pairs:
        pair = i.split(': ')
        if len(pair) < 2:
            continue
        # Other symbols from the key-value pair should be stripped.
        dictionary[pair[0].strip('\'\'\"\"')] = pair[1].strip('\'\'\"\"')
    return dictionary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, help="A dictionary contains conditions that the experiment results need to fulfill (e.g., tag, task_name, few_shot_type)")
    
    # These options should be kept as their default values
    parser.add_argument("--log_path", type=str, default="output/multitask/meta-llama_Llama-2-7b-hf/", help="Log path.")
    parser.add_argument("--log", type=str, default="log_summary.json", help="Log file.")
    # parser.add_argument("--key", type=str, default='', help="Validation metric name")

    args = parser.parse_args()

    condition = eval(args.condition)

    

    ## load result
    result_list = []
    with open(os.path.join(args.log_path, args.log)) as f:
        for line in f:
            result_list.append(parse_str_to_dict(line))

    ## set up key
    key = 'accuracy'
    key2 = ""
    if condition['task'].startswith('glue-cola'):
        key = 'mcc'
    elif condition['task'].startswith('glue-qqp'):
        key2 = 'F1'
    else:
        pass
    

    seed_result = {}
    seed_best = {}
    # print(result_list)

    for item in result_list:
        ok = True
        for cond in condition:
            if isinstance(condition[cond], list):
                if cond not in item or (item[cond] not in condition[cond]):
                    ok = False
                    break
            else:
                # print(item)
                # print("zhuoyan ", cond, cond not in item)
                # print("zhuoyan ", cond,item[cond] != condition[cond])
                # Check if the condition value is an integer, if so, convert the item value to an integer for comparison
                item_value = int(item[cond]) if isinstance(condition[cond], int) else item[cond]
                if cond not in item or (item_value != condition[cond]):
                    # print("break on this one")
                    ok = False
                    break
        # print("zhuoyan OK", ok)
        if ok:
            seed = str(item['seed'])
            if seed not in seed_result:
                seed_result[seed] = [item]
                seed_best[seed] = item
            else:
                seed_result[seed].append(item)
                if item[key] > seed_best[seed][key]:
                    seed_best[seed] = item

    # print("zhuoyan==========================================================================")
    # print(seed_result)
    # print(seed_best)
    # print("zhuoyan==========================================================================")

    print("evaluate on task {}".format(condition['task']))
    final_result_test = np.zeros((len(seed_best)))
    if len(key2) > 0:
        final_result_test2 = np.zeros((len(seed_best)))
    for i, seed in enumerate(seed_best):
        final_result_test[i] = seed_best[seed][key]
        print("{}: best test ({}: {:.5f}) | total trials: {}".format(
            seed,
            key,
            float(seed_best[seed][key]),
            len(seed_result[seed])
        ))
        if len(key2) > 0:
            final_result_test2[i] = seed_best[seed][key2]
            print("{}: best test ({}: {:.5f}) | total trials: {}".format(
            seed,
            key2,
            float(seed_best[seed][key2]),
            len(seed_result[seed])
        ))
        s = ''
        for k in ['num_fewshot', 'test_batch_size']:
            s += '| {}: {} '.format(k, seed_best[seed][k])
        print('    ' + s)

    # print("zhuoyan==========================================================================")
    # print(final_result_test)
    # print("zhuoyan==========================================================================")
    s = "mean +- std: {:.2f} ({:.2f}) (median {:.2f})".format(final_result_test.mean() * 100, final_result_test.std() * 100, np.median(final_result_test) * 100)
    if len(key2) > 0:
        s += "    key2: {} | mean +- std: {:.2f} ({:.2f}) (median {:.2f})".format(key2, final_result_test2.mean() * 100, final_result_test2.std() * 100, np.median(final_result_test2) * 100)
    print(s)

    with open(os.path.join(args.log_path,"mean.txt"), 'a') as f:
        f.write(str(condition['task']) + "," + str(condition['num_fewshot']) + "," + str(final_result_test.mean() * 100) + '\n')
    
    # if not os.path.exists(os.path.join(args.log_path,"mean.txt")):
    #     with open(os.path.join(args.log_path,"mean.txt"), 'a') as f:
    #         f.write(str(condition['task']) + "," + str(condition['num_fewshot']) + "," + str(final_result_test.mean() * 100) + '\n')
    # else:
    #     print("exist, skip")

if __name__ == '__main__':
    main()
