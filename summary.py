import os
import numpy as np
 
report_key = "acc"

folder_name_list = [
    #"gpt2-large",  
    #"llama-7b",
    #"llama-2-7b",  
    #"llama-13b",
    #"llama-2-13b",  
    #"llama-30b" 
    #"llama-65b", 
    #"llama-2-70b" 
    #"llama-2-7b-chat", 
    #"llama-2-13b-chat", 
    "llama-2-70b-chat" 
]

data_list = [
        #"glue-mrpc",
        "glue-rte",
        "glue-sst2",
        "glue-qqp",
        #"tweet_eval-hate",
        #"openbookqa",
        #"commonsense_qa",
        #"superglue-copa"
        ]

root_path = "out"

def read_file(n):
    f = open(n, 'r')
    s = f.read()
    f.close()
    nums = s.split("\n")
    return float(nums[0]),float(nums[1])

results = {}
for f in folder_name_list:
    results[f]={}
    for d in data_list:
        results[f][d]={}
    folder_path = os.path.join(root_path, f)
    files = os.listdir(folder_path)
    acc_files = []
    for n in files:
        if n[:3] == "acc":
            acc_files.append(n)
    
    result_dic = {}
    for n in acc_files:
        acc, f1 = read_file(os.path.join(folder_path, n))
        if "s=" in n:
            key = n.split("s=")[0] + "s="
        else:
            key = n
        if key not in result_dic:
            result_dic[key] = {}
            result_dic[key]["acc"] = []
            result_dic[key]["f1"] = []
        result_dic[key]["acc"].append(acc)
        result_dic[key]["f1"].append(f1)
    
    keys = sorted(result_dic.keys())
    for key in keys:
        print(len(result_dic[key][report_key]))
        print(key)
        #if "s=" in key:
        #    assert len(result_dic[key]["acc"]) == 5
        #else:
        #    assert len(result_dic[key]["acc"]) == 1
        result_dic[key]["acc"]=np.mean(np.array(result_dic[key]["acc"]))
        result_dic[key]["f1"]=np.mean(np.array(result_dic[key]["f1"]))
        for d in data_list:
            if d in key:
                if "txt" in key:
                    results[f][d]["no     "]=result_dic[key][report_key]
                elif "permutated" in key:
                    results[f][d]["permute"]=result_dic[key][report_key]
                elif "random" in key:
                    results[f][d]["random "]=result_dic[key][report_key]
                else:
                    results[f][d]["gold   "]=result_dic[key][report_key]
                break

method = ["no     ", "gold   ", "random ", "permute"]
for f in folder_name_list:
    print("\n\n",f)
    for m in method:
        for d in data_list:
            if m in results[f][d].keys():
                print("{0:.2f}".format(results[f][d][m]), end=" ")
        print()



if __name__=='__main__':
    pass
