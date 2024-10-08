import json
import jsonlines
import os
from config import getConfig
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
from peft import get_peft_model
from peft import LoraConfig, PeftConfig
from datasets import Dataset
from dataloader import DataForLoRA
from torch.utils.data import DataLoader
args = getConfig()

class LoRA:
    def __init__(self, args):
        self.args = args
        
        path = str(args.batch_size) + "batch_" + str(args.epoch) + "epoch" 
        self.output_path = f"../results/{path}"
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load Data
        tr_ds = DataForLoRA(args.tr_data_path, args)
        tr_ds = tr_ds.preprocess(mode="train")
        self.tr_ds = Dataset.from_list(tr_ds)
        
        vl_ds = DataForLoRA(args.vl_data_path, args)
        vl_ds = vl_ds.preprocess(mode="val")
        self.vl_ds = Dataset.from_list(vl_ds)

        print(f'LoRA Train Data : {len(self.tr_ds)}개')
        print(f'LoRA Val Data : {len(self.vl_ds)}개')
        print('-'*75)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            padding_side='right',
        )

        print(self.tokenizer)
        print('-'*75)
        
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=args.lora_target_modules,
            task_type='CAUSAL_LM',
        )

        print(peft_config)
        print('-'*75)

        # Model for PEFT (LoRA, P-Tuning, ...)
        self.model = get_peft_model(
            self.model,
            peft_config,
        )

        print(self.model)
        print('-'*75)

        print(self.model.config)
        print('-'*75)

        # Trainable Parameters
        self.model.print_trainable_parameters()
        print('-'*75)

    def peft(self):
        print(self.tr_ds['text'][0])
        print('-'*75)

        # collator = None
        # if self.args["is_completion_only"]:
        instruction_template = "\n<|im_start|>user"
        response_template = "\n<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=self.tokenizer,
        )

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_path,
            learning_rate=self.args.lr,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epoch,
            weight_decay=0.01,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps = 50,
            logging_steps=50,
            load_best_model_at_end=True
        )

        print(training_args)
        print('-'*75)

        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            max_seq_length=self.args.max_len,
            train_dataset=self.tr_ds,
            eval_dataset=self.vl_ds,
            dataset_text_field='text',
            data_collator=collator,
            args=training_args,
        )

        trainer.train()

        trainer.save_model(self.output_path)
        trainer.model.config.save_pretrained(self.output_path)

if __name__ == '__main__':
    lora = LoRA(args)
    lora.peft()

