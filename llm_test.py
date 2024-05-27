# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import json
import logging
import numpy as np

from collections import Counter

from llm_data import MetaICLData
from llm_model import MetaICLModel

from utils.data import load_data

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)
    
    add_newlines = not args.base_model_path.startswith("gpt2")
    
    metaicl_model = MetaICLModel(args.base_model_path, logger, args.out_dir, num_gpus=args.num_gpus)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        if args.do_zeroshot:
            max_length = min(max_length * args.k, 1024)
        else:
            max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, args.base_model_path, args.method,args.use_demonstrations, args.k,
                               max_length, max_length_per_example)

    results = []
    errors = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:

        ### data ...
        train_data = load_data(args.task, "train", args.k, seed=seed, config_split=config_split,
                               datasets=None if args.dataset is None else args.dataset.split(","))
        dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), is_null=args.is_null)

        if args.use_random_english_words:
            from english_words import english_words_set
            english_words_set = sorted(english_words_set)
            np.random.seed(int(seed))

        train_counter = Counter()
        dev_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for dp in dev_data:
            dev_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        for k, v in dev_counter.items():
            logger.info("[Dev] %s\t%d" % (k, v))

        logger.info("%s on %s (%d train, %d dev)" % (args.method, args.task, len(train_counter), len(dev_counter)))

        for test_task in dev_counter:
            curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            curr_train_data = [dp for dp in train_data if dp["task"]==test_task]
            assert len(curr_dev_data)>0
            assert not args.use_demonstrations or len(curr_train_data)==args.k, \
                    (args.use_demonstrations, len(curr_train_data), args.k)

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"]=="classification"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

            if args.use_random_english_words:
                # create a mapping
                options = curr_dev_data[0]["options"]
                mapping = {option: np.random.choice(english_words_set) for option in options}
                new_options = list(mapping.values())
                for dp_idx, dp in enumerate(curr_train_data):
                    assert dp["output"] in options, (dp, options)
                    curr_train_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_train_data[dp_idx]["options"] = new_options
                for dp_idx, dp in enumerate(curr_dev_data):
                    assert dp["output"] in options, (dp, options)
                    curr_dev_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_dev_data[dp_idx]["options"] = new_options

            result = run(logger, test_task, metaicl_data, metaicl_model,
                         curr_train_data, curr_dev_data, seed, is_classification, add_newlines)

            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                results.append(result)

    if args.is_null:
        return

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100*np.mean(results)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed, is_classification, add_newlines):

    split_name = args.split
    if args.is_null:
        split_name += "-null"
    cache_path = os.path.join(args.out_dir,task, # add one more sub-dir to make *.pkl and *txt files separated clear
                                "{}-{}-{}{}{}{}{}.pkl".format(
                                    task,
                                    split_name,
                                    metaicl_data.method,
                                    "-k={}".format(args.k) if args.use_demonstrations else "",
                                    "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                    "" if add_newlines else "-no-newlines",
                                    "-randomEnglish" if args.use_random_english_words else ""))
    if not os.path.exists(os.path.join(args.out_dir,task)):
        os.makedirs(os.path.join(args.out_dir,task),exist_ok=True)
    metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    metaicl_data.print_tensorized_example()
    assert False
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    
    file_names = prediction_path.split("/")
    file_names[-1] = "acc-"+file_names[-1]
    acc_path = "/".join(file_names)
    
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")
        acc_path = acc_path.replace(".txt", "-calibrated.txt")

    if os.path.exists(prediction_path):
        return 0

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        if metaicl_model.is_none():
            metaicl_model.load()
            metaicl_model.eval()

        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)

    assert len(losses)==len(metaicl_data)

    if args.is_null:
        return None

    if args.use_calibration:
        assert args.do_zeroshot
        bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
        assert os.path.exists(bias_path), bias_path
        with open(bias_path, "rb") as f:
            bias_losses = pkl.load(f)

        losses = np.array(losses)
        bias_losses = np.array(bias_losses)
        assert losses.shape == bias_losses.shape
        losses -= bias_losses

    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in dev_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf[0])

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    # with open(acc_path, "w") as f: 
    #     f.write("%s \n" % perf[0])
    #     f.write("%s \n" % perf[1])
    
    # added by zhuoyan, log value summary
    values_summary = {}
    values_summary["task"] = task
    values_summary["seed"] = seed
    values_summary['num_fewshot'] = args.k if args.use_demonstrations else 0
    values_summary['new_lines'] = True if add_newlines else False
    values_summary['split_name'] = split_name
    values_summary['test_batch_size'] = args.test_batch_size
    values_summary['accuracy'] = perf[0]
    values_summary['F1'] = perf[1]

    # print(values_summary)

    log_path = args.out_dir
    # print("write values to ", os.path.join(log_path, "log_summary.json"))
    with open(os.path.join(log_path, "log_summary.json"), 'a') as f:
        f.write(str(values_summary) + '\n')

    return perf[1]

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100")

    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--base_model_path", type=str, required=True)

    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    # log_file_dir = os.path.dirname(args.log_file)
    # os.makedirs(log_file_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
