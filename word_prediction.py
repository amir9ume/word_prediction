import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import os
import logging

from data_read import Data_Preprocess
from model import MCQ_Filler

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere


path_train_data='./data/train.csv'
model_save_checkpoint='./data/ft_xlm_model'

if (os.path.exists('./data/coveo_train_data.pt')==False) or (os.path.exists('./data/coveo_val_data.pt')==False):
    print('Tokenizing text and preprocessing...')
    tokenized_text_input_ids,tokenized_text_attention_mask,tokenized_choices,targets= Data_Preprocess(path_train_data,train_flag=True).tokenize_preprocess()
    coveo_data= TensorDataset(tokenized_text_input_ids,tokenized_text_attention_mask,tokenized_choices,targets)

    train_size = round(0.80 * len(coveo_data))
    val_size= round(0.20* len(coveo_data))
    train_dataset, val_dataset = random_split(coveo_data, [train_size, val_size])
    print('Data pre-processing complete.')

    torch.save(train_dataset,'./data/coveo_train_data.pt')
    torch.save(val_dataset,'./data/coveo_val_data.pt')
else:
    print('Train data already exists...loading')
    train_dataset=torch.load('./data/coveo_train_data.pt')
    val_dataset=torch.load('./data/coveo_val_data.pt')
    print('Train data loaded')

batch_size=64
train_dataloader = DataLoader( train_dataset, sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

validation_dataloader = DataLoader(
            val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size 
        )

loss=torch.nn.CrossEntropyLoss()
model= MCQ_Filler().to(device)
optimizer = AdamW(model.parameters(),lr = 2e-5, eps = 1e-8 )

epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

for epoch_i in range(0, epochs):
    model.train()
    total_train_loss = 0
    total_correct=0
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('Batch {:>5,}  of  {:>5,}     .'.format(step, len(train_dataloader)))
            print('loss : ',total_train_loss/(step))
            print("accuracy :  {0:.2f}".format(total_correct/(step * batch_size)))    
            
            if os.path.exists(model_save_checkpoint):
                os.remove(model_save_checkpoint)
            torch.save(model.state_dict(),model_save_checkpoint)
            print('model saved at step')

        tk_text_input_ids=batch[0].to(device)
        tk_text_attn_mask=batch[1].to(device)
        tk_choice=batch[2].to(device)
        target=batch[3].to(device)

        model.zero_grad() 
        mask_token_index = torch.where(tk_text_input_ids ==250001)[1]
        if mask_token_index.shape[0]==batch_size:
            prediction= model(tk_text_input_ids,tk_text_attn_mask,tk_choice
            ,mask_token_index)
            loss_batch= loss(prediction,target.argmax(dim=1).long())
            
            loss_batch.backward()
            total_train_loss+= loss_batch.item()
            
            correct=torch.sum(target.argmax(dim=1)==F.softmax(prediction,dim=1).argmax(dim=1))
            total_correct+=correct.item()
            
            optimizer.step()
            scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)  
    accuracy_train= total_correct/(len(train_dataloader) * batch_size)
    print("Epoch:", epoch_i,"-Average training loss: {0:.2f}".format(avg_train_loss))
    print("Epoch:", epoch_i,"-Average training accuracy:  {0:.2f}".format(accuracy_train))
    model.eval()

    
    total_validation_loss=0
    total_val_correct=0
    
    with torch.no_grad(): 
        for batch in validation_dataloader:  
            tk_text_input_ids=batch[0].to(device)
            tk_text_attn_mask=batch[1].to(device)
            tk_choice=batch[2].to(device)
            target=batch[3].to(device)

            mask_token_index = torch.where(tk_text_input_ids ==250001)[1]
            if mask_token_index.shape[0]==batch_size:
                prediction= model(tk_text_input_ids,tk_text_attn_mask,tk_choice,mask_token_index)
                loss_val_batch= loss(prediction,target.argmax(dim=1).long())
                total_validation_loss+=loss_val_batch.item()
                val_correct=torch.sum(target.argmax(dim=1)==F.softmax(prediction,dim=1).argmax(dim=1))
                total_val_correct+=val_correct.item()
        
        avg_val_loss= total_validation_loss/ len(validation_dataloader)    
        accuracy_val= total_val_correct/(len(validation_dataloader)* batch_size)
        print(" Average validation loss: {0:.2f}".format(avg_val_loss))
        print("Average validation accuracy:  {0:.2f}".format(accuracy_val))

print('Saving model ..... :')
trained_model_path='./data/final_xlm_model'
torch.save(model.state_dict(),trained_model_path)

