import os
import torch
from typing import Union, Tuple, Callable
from torch.utils.data import DataLoader
import torch.nn as nn


class BaseValidator:
    def __init__(self, model: nn.Module,
                 val_dataloader: DataLoader,
                 device= torch.device("cpu"),
                 logs_root: str = os.getcwd(),
                 ):
        self.model_ft = model.to(device).eval()# get_model(model_name, model_root, device).eval()
        self.device = device
        self.val_dataloader = val_dataloader # get_testloaders(testloader_path)
        self.logs_root = logs_root

    
    def forward(self, data: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            outputs = self.model_ft(x)
        
        return (y, outputs)
    
    
    def val_loop(self, dataloader: DataLoader):
        targets_list = []
        preds_list = []
        
        for data in dataloader:
            (targets, preds) = self.forward(data)
            
            if targets.shape:
                targets_list.extend(targets)
                preds_list.extend(preds)
                
        out_lists = [torch.stack(targets_list).cpu().squeeze(),
                     torch.stack(preds_list).cpu().squeeze()]
        
        return out_lists

    
    def test(self, all: bool=False):
        if not all:
            data = next(iter(self.val_dataloader))
            return self.forward(data)
        else:
           return self.val_loop(self.val_dataloader)
    
        