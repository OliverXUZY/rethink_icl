# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM

# from llama_parallama import LLaMAPolicy
# from parallelformers import parallelize

class MetaICLModel(object):

    def __init__(self, base_model_path, logger=None, out_dir=None, fp16=True, num_gpus=1):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.base_model_path = base_model_path
        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        # self.num_gpus = num_gpus

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        self.n_gpu = n_gpu
        self.device = device

        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus" % (self.device, self.n_gpu)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''
        # self.model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                device_map="auto")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #         self.base_model_path, low_cpu_mem_usage = True)
        self.model.eval()
        # parallelize(self.model, num_gpus=self.num_gpus, fp16=self.fp16, verbose='detail', custom_policies=[LLaMAPolicy])



    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        # if verbose:
        dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].to(0)
            attention_mask=batch[1].to(0)
            token_type_ids=batch[2].to(0)
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].to(0)
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
