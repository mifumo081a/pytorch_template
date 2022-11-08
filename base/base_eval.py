import os
import torch
from typing import Union, Tuple, Callable
from torch.utils.data import DataLoader
import torch.nn as nn


class BaseEvaluator:
    def __init__(self, model: nn.Module,
                 dataloaders: Callable[[], DataLoader],
                 testloaders: Callable[[], DataLoader],
                 device= torch.device("cpu"),
                 logs_root: str = os.getcwd(),
                 ):
        self.model_ft = model.to(device).eval()# get_model(model_name, model_root, device).eval()
        self.device = device
        self.dataloaders = dataloaders # get_dataloaders(dataloader_path)
        self.testloaders = testloaders # get_testloaders(testloader_path)
        self.logs_root = logs_root

    
    def forward(self, data: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            outputs = self.model_ft(x)
        
        return (y, outputs)
    
    
    def calc_dataloader(self, dataloader: DataLoader):
        targets_list = []
        preds_list = []
        
        for data in dataloader:
            (targets, preds) = self.forward(data)
            
            if targets.shape:
                targets_list.extend(targets)
                preds_list.extend(preds)
                
        cls_lists = [torch.stack(targets_list).cpu().squeeze(),
                     torch.stack(preds_list).cpu().squeeze()]
        
        return cls_lists

    
    def test(self, random: bool=False, all: bool=False):
        is_random = ["test", "randomtest"][int(random)]

        if not all:
            data = next(iter(self.testloaders[is_random]))
            return self.forward(data)
        else:
           return self.calc_dataloader(self.testloaders["test"])
    
        