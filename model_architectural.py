
import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import json
#from tensorflow.keras import layers , activations , models , preprocessing, utils
#from tqdm.autonotebook import get_ipython
from transformers import AutoModel, BertTokenizerFast
from torch import save
import warnings
warnings.filterwarnings("ignore")



bert = AutoModel.from_pretrained('bert-base-uncased')
class BERT_Arch(nn.Module):
    def __init__(self, bert):      
        super(BERT_Arch, self).__init__()
        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.2)
      
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,19)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)
        #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model  
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
      
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)
   
        # apply softmax activation
        x = self.softmax(x)
        return x
# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False



