import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from data_read import Data_Preprocess
from model import MCQ_Filler
import pandas as pd

torch.manual_seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_run(test_file_path,path_model,device):
    
    saved_data_location='./data/coveo_test_data.pt'
    if os.path.exists(saved_data_location)==False:
        
        print('Tokenizing text and preprocessing...')
        tokenized_text_input_ids,tokenized_text_attention_mask,tokenized_choices, sample_indices= Data_Preprocess(test_file_path,train_flag=False).tokenize_preprocess()
        coveo_data= TensorDataset(tokenized_text_input_ids,tokenized_text_attention_mask,tokenized_choices,sample_indices)
        print('Data pre-processing complete.')
        torch.save(coveo_data,saved_data_location)
    else:
        print('Test data already exists...loading')
        coveo_data=torch.load(saved_data_location)
        
    batch_size=1
    test_dataloader = DataLoader(
                coveo_data, sampler = SequentialSampler(coveo_data), batch_size = batch_size 
            )

    loss=torch.nn.CrossEntropyLoss()
    model= MCQ_Filler().to(device)
    if os.path.exists(path_model):
        model.load_state_dict(torch.load(path_model))
        print('model loaded')
    model.eval()

    indices=[]
    results=[]
    with torch.no_grad(): 
        for batch in test_dataloader:  
            tk_text_input_ids=batch[0].to(device)
            tk_text_attn_mask=batch[1].to(device)
            tk_choice=batch[2].to(device)
            sample_idx= batch[3]

            mask_token_index = torch.where(tk_text_input_ids ==250001)[1]
            prediction= F.softmax(model(tk_text_input_ids,tk_text_attn_mask,tk_choice,mask_token_index),dim=1)
            pred_values= prediction.to(torch.float16)            

            indices.append(int(sample_idx.item()))
            results.append(pred_values.detach().cpu().numpy()[0])
   
    pd.set_option('precision', 4)
    
    frame = pd.DataFrame(data=results, columns=["choice1","choice2","choice3","choice4","choice5","choice6"])
    id=pd.DataFrame(data=indices,columns=['idx'])
    df_test_results= pd.concat([id,frame],axis=1)
    print(df_test_results.sample(5))
    df_test_results.to_csv('results.csv',index=None)

testing=True
if testing:
    path_to_model='./data/final_xlm_model'
    path_to_test_data= './data/test.csv'

    test_run(path_to_test_data,path_to_model,device=device)
    print('Test run complete')
