import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset

## needs TO-DO /CLEANUP!

class CardiacDataset(Dataset):
    def __init__(self, data_dir = "BrachioLab/cardiac-timeseries", config_path = "BrachioLab/cardiac-prediction", split: str = "test"):
        self.dataset = load_dataset(data_dir, split = split)
        self.config = InformerConfig.from_pretrained(config_path)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_dataloader(self, batch_size = 5, compute_loss = True, shuffle = True):
        return create_test_dataloader_raw(
            dataset=self,
            batch_size=batch_size,
            compute_loss=compute_loss,
            shuffle=shuffle
        )

            

########################
# FYI - 
# the following has no alarms
# CSN_suffix = '000' 
# CSN_suffix = '001' 
# CSN_suffix = '002' 


