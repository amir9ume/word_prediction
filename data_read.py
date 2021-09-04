import torch

import pandas as pd
import numpy as np

from transformers import  AutoTokenizer


class Data_Preprocess:
    def __init__(self, file_path, train_flag=False):
            '''
            init takes file path and train flag true or false
            
            Send file path to load train or test set
            Set train_flag to True for training. Default is False
            Choice 5,6 columns corrupt on both Train and Test set.
            Replacing those NaN with <pad> tokens
            '''
            self.df= pd.read_csv(file_path).sample(frac=1)
            self.train_flag= train_flag
            
            self.df['choice5'] = self.df['choice5'].replace(np.nan, '<pad>')
            self.df['choice6'] = self.df['choice6'].replace(np.nan, '<pad>')
        
            self.labels_one_hot= torch.zeros(len(self.df.index),6)        
            self.tokenizer= AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __len__(self):
        return len(self.df.index)

    
    def __getitem__(self, idx):
        row_data= self.df.iloc[idx]
        text_data = row_data['text'].replace('[BLANK]','<mask>').lower() 
        choices= list(row_data[2:8].values)
        tokenized_choices= self.tokenizer(choices,add_special_tokens=False,
                            return_attention_mask=False,return_token_type_ids=None,
                            max_length=1,truncation=True,return_tensors='pt')['input_ids'].squeeze(dim=1)
    
        if self.train_flag:
            label = row_data['label']
            z=int(label.split('choice')[1])-1
            self.labels_one_hot[idx][z]=1
            target=self.labels_one_hot[idx]
            return text_data, tokenized_choices, target
        else:
            return text_data, tokenized_choices, torch.tensor(row_data['idx'])
        

    def tokenize_preprocess(self):
        size= self.__len__()
        all_texts=[]
        tokenized_choices=[]
        targets=[]
        sample_indices=[]

        for i in range(size):
            if self.train_flag:
                t,c,target=self.__getitem__(i)
                if (c.shape[0]!=6) or ('<mask>' not in t):
                    continue
                targets.append(target)
                all_texts.append(t)
                tokenized_choices.append(c)
            else:
                t,c, idx=self.__getitem__(i)
                if (c.shape[0]!=6) or ('<mask>' not in t):
                    continue
                sample_indices.append(idx)
                all_texts.append(t)
                tokenized_choices.append(c)
            
        tokenized_text=self.tokenizer(all_texts,padding="max_length",max_length=512,truncation=True, return_tensors="pt" )
        tokenized_text_input_ids= tokenized_text['input_ids']
        tokenized_text_attention_mask= tokenized_text['attention_mask']

        if self.train_flag:
            return tokenized_text_input_ids,tokenized_text_attention_mask, torch.stack(tokenized_choices,dim=0), torch.stack(targets,dim=0)    
        else:
            return tokenized_text_input_ids,tokenized_text_attention_mask, torch.stack(tokenized_choices,dim=0), torch.stack(sample_indices,dim=0)

