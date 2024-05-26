from datasets import load_dataset
import pandas as pd
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from intermediate_dual_clustering.dataset import make_dataset
from intermediate_dual_clustering.trainer import test

def load_hf_data(path):

    hf_data = load_dataset(path)
    hf_data_df = pd.DataFrame(hf_data['train'])

    hf_data_dict = dict()
    hf_data_list = list()

    for instruction, output, input in zip(hf_data_df['instruction'], hf_data_df['output'], hf_data_df['input']):
        instruction = f"{instruction}\n{input}"
        
        hf_data_dict = {'target' : f"instruction : {instruction}\noutput : {output}"}
        hf_data_list.append(hf_data_dict)

    return hf_data_list
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', required = True)
    parser.add_argument('--hf_tokenizer_path', required = True)
    parser.add_argument('--model_path', required = True)
    args = parser.parse_args()

    model = nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)
    test_data = load_hf_data(args.test_data_path)
    test_data = make_dataset(test_data, tokenizer, model.module.config.max_position_embeddings)
    test_dataloader = DataLoader(test_data,
                                 batch_size = 64,
                                 collate_fn = test_data.collate_fn
                                )
    
    inputs_list, preds_list = test(
                                test_dataloader,
                                model,
                                )

    task = [] # 설정

    
    predict_dict = dict()
    predict_list = list()
    
    for inputs, preds in tqdm(zip(inputs_list, preds_list), desc = 'make predict file ...'):
        predict_dict = { 'inputs' : inputs,
                        'preds' : task[preds] }
        predict_list.append(predict_dict)

    predict_df = pd.DataFrame(predict_list)
    predict_df.to_csv('predict.csv', index = False)
    