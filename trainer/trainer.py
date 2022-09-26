import os
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, Adam, SGD
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Tuple, Union
from ..base import BaseTrainer
from ..pickle_io import *
from sklearn.metrics import accuracy_score


class Trainer_Classifier(BaseTrainer):
    def __init__(self, model: nn.Module, device: torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 optim_init: Callable[[], Optimizer]=Adam,
                 lr=1e-3, epochs=10):
        super().__init__(model=model, device=device,
                         dataloaders=dataloaders,
                         optim_init=optim_init, lr=lr, epochs=epochs)
        

    def step_func(self, data):
        loss_func = nn.CrossEntropyLoss()
        x, labels = data
        x, labels = x.to(self.device), labels.to(self.device)

        if self.model_ft.training:
          self.optimizer.zero_grad()
          outputs, _ = self.model_ft(x)
          loss = loss_func(outputs, labels)
          _, predictions = torch.max(outputs, 1)
          loss.backward()
          self.optimizer.step()
        else:
          with torch.no_grad():
            outputs, _ = self.model_ft(x)
            loss = loss_func(outputs, labels)
            _, predictions = torch.max(outputs, 1)
        
        return loss.item(), (labels, predictions)


    def epoch_func(self) -> Union[float, Tuple[float, Optional[Dict[str, float]]]]:
        """
        1エポックでの処理をする。
        データローダーを繰り返し処理する部分を書く
        """
        dataset_size = len(self.dataloaders[self.phase].dataset)
        running_loss = 0.0

        targets = []
        preds = []

        for data in self.dataloaders[self.phase]:
            loss, *eval_var = self.step_func(data=data)

            running_loss += loss * data[0].size(0)

            if len(eval_var):
                cls_eval = eval_var[0]

                # classification
                targets.extend(cls_eval[0])
                preds.extend(cls_eval[1])
                
        epoch_loss = running_loss / dataset_size

        # classification
        targets = torch.stack(targets).cpu()
        preds = torch.stack(preds).cpu()
        cls_epoch_acc = accuracy_score(targets, preds)
        # cls_epoch_acc = torch.sum(targets_cls == preds_cls) / preds_cls.shape[0]

        return epoch_loss, {"Acc": cls_epoch_acc.item()}  

    
class LS_Trainer(BaseTrainer):
    def __init__(self, model: nn.Module, device: torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 optim_init: Callable[[], Optimizer]=Adam,
                 lr=1e-3, epochs=10):
        super().__init__(model=model, device=device,
                         dataloaders=dataloaders,
                         optim_init=optim_init, lr=lr, epochs=epochs)
    
    
    def step_func(self, data):
        x, labels = data
        x, labels = x.to(self.device), labels.to(self.device)
        
        m, n = x.shape[-2:]
        ls = 1./np.sqrt(max(m, n))
        ln = 1.
        nucloss = lambda x: torch.sum(torch.linalg.svdvals(torch.nan_to_num(x)))*ln
        l1loss = lambda x: torch.sum(torch.abs(x)*ls)
        
        if self.model_ft.training:
          self.optimizer.zero_grad()
          outputs, _ = self.model_ft(x)
          loss = l1loss(outputs) + nucloss(x-outputs)
          loss.backward()
          self.optimizer.step()
        else:
          with torch.no_grad():
            outputs, _ = self.model_ft(x)
            loss = l1loss(outputs) + nucloss(x-outputs)
        
        return loss.item()


    def epoch_func(self) -> Union[float, Tuple[float, Optional[Dict[str, float]]]]:
        """
        1エポックでの処理をする。
        データローダーを繰り返し処理する部分を書く
        """
        dataset_size = len(self.dataloaders[self.phase].dataset)
        running_loss = 0.0

        for data in self.dataloaders[self.phase]:
            loss = self.step_func(data=data)

            running_loss += loss * data[0].size(0)
                
        epoch_loss = running_loss / dataset_size

        return epoch_loss, {}