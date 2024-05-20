from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import re
import numpy as np
import pandas as pd
import random

seed = 2610
random.seed(seed)
np.random.seed(seed)

def compute_metrics(eval_pred):
    y_pred, y_true = np.argmax(eval_pred.predictions, -1), eval_pred.label_ids
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)}

def data_cleaning(data, max_char=15000):
    comment_regex = r'(//[^\n]*|\/\*[\s\S]*?\*\/)'
    newline_regex = '\n{1,}'
    whitespace_regex = '\s{2,}'
    
    def replace(inp, pat, rep):
        return re.sub(pat, rep, inp)
    data['truncated_code'] = (data['code'].apply(replace, args=(comment_regex, ''))
                                        .apply(replace, args=(newline_regex, ' '))
                                        .apply(replace, args=(whitespace_regex, ' '))
                            )
    # remove all data points that have more than 15000 characters
    length_check = np.array([len(x) for x in data['truncated_code']]) > max_char
    data = data[~length_check]
    return data

def to_huggingface_dataset(tokenizer, data_train, data_test, data_valid):
    dts = DatasetDict()
    dts['train'] = Dataset.from_pandas(data_train)
    dts['test'] = Dataset.from_pandas(pd.concat([data_test, data_valid]))
    dts['valid'] = Dataset.from_pandas(pd.concat([data_test, data_valid]))
    def tokenizer_func(examples):
        result = tokenizer(examples['truncated_code'])
        return result

    dts = dts.map(tokenizer_func,
                batched=True,
                batch_size=4
                )
    dts.set_format('torch')
    dts.rename_column('label', 'labels')
    dts = dts.remove_columns(['code', 'truncated_code', '__index_level_0__'])
    return dts

def train_test_valid_split(data, tokenizer, train_size=0.8, test_size=0.1, valid_size=0.1):
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(data.loc[:, data.columns != 'label'],
                                                                data['label'],
                                                                train_size=train_size,
                                                                stratify=data['label']
                                                               )
    test_size /= (test_size+valid_size)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid.loc[:, X_test_valid.columns != 'label'],
                                                        y_test_valid,
                                                        test_size=test_size,
                                                        stratify=y_test_valid)
    data_train = X_train
    data_train['label'] = y_train
    data_test = X_test
    data_test['label'] = y_test
    data_valid = X_valid
    data_valid['label'] = y_valid
    
    dts = to_huggingface_dataset(tokenizer, data_train, data_test, data_valid)
    return dts

def data_preprocessing(file_path="full_data.csv",
                       model_ckpt='neulab/codebert-c'):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    data = pd.read_csv(file_path)
    data = data_cleaning(data)
    dts = train_test_valid_split(data, tokenizer)
    
    return dts
    
    