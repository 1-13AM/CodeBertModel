import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from model import CodeBertModel
from utils import *
import pandas as pd
import numpy as np
import random
import os
import wandb

wandb.login()
seed = 2610
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ckpt = os.environ.get('MODEL_CKPT', 'neulab/codebert-c')
n_attn_head = int(os.environ.get('N_ATTN_HEAD', 2))
print('-' * 80)
print(f"Your model is using {model_ckpt} with {n_attn_head} attention heads")
print('-' * 80)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = CodeBertModel(model_ckpt=model_ckpt, n_attn_head=n_attn_head)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dts = data_preprocessing(model_ckpt=model_ckpt)

training_arguments = TrainingArguments(output_dir = 'codebertmodel',
                                      evaluation_strategy = 'epoch',
                                      per_device_train_batch_size = 1,
                                      per_device_eval_batch_size = 1,
                                      gradient_accumulation_steps = 24,
                                      learning_rate = 2e-5,
                                      num_train_epochs = 3,
                                      warmup_ratio = 0.1,
                                      lr_scheduler_type = 'cosine',
                                      logging_strategy = 'steps',
                                      logging_steps = 10,
                                      save_strategy = 'no',
                                      fp16 = True,
                                      metric_for_best_model = 'recall',
                                      optim = 'adamw_torch',
                                      report_to = 'none'
                                      )

trainer = Trainer(model=model,
                  data_collator=data_collator,
                  args=training_arguments,
                  train_dataset=dts['train'],
                  eval_dataset=dts['valid'],
                  compute_metrics=compute_metrics,
                 )

if __name__ == '__main__':

    wandb.init(project='huggingface', entity='ashton_h', name=f'{model_ckpt}-nhead-{n_attn_head}')

    trainer.train()
    check = trainer.predict(dts['test'])
    compute_metrics(check)
    
    wandb.finish()
