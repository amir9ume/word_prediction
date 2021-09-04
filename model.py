import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM, AutoTokenizer

class MCQ_Filler(nn.Module):
    def __init__(self, device='cpu'):
        super(MCQ_Filler,self).__init__()
        '''
        Initialization
        
        XLM-RoBERTa was trained on 2.5TB 
        of newly created clean CommonCrawl 
        data in 100 languages. 

        Model gradients only backpropagated on the final layer
        LM head, due to compute and time limitations
        '''
        self.xlm_model= AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        self.tokenizer= AutoTokenizer.from_pretrained("xlm-roberta-base")

        for i,param in enumerate(self.xlm_model.parameters()):
            if (i>=199):
                param.requires_grad= True
            else:
                param.requires_grad = False


    def forward(self,input_ids,attention_masks,tokenized_choices,mask_token_index):        
        token_logits = self.xlm_model(input_ids,attention_masks,return_dict=True).logits        
        mask_token_logits = token_logits[0, mask_token_index, :]
        only_target_choice_logits= mask_token_logits.gather(1,tokenized_choices)
        
        return only_target_choice_logits
        
    