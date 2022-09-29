import os
import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
from typing import Dict, Tuple, Union, Optional, OrderedDict, Callable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from ..utils import get_optimizer
from ..pickle_io import save_as_pickle


class BaseTrainer:
    def __init__(self, model: nn.Module, 
                 device= torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 optim_init: Callable[[], Optimizer]=SGD,
                 lr=1e-3, epochs=10):
        self.model_init = model # get_model("nontrain", model_root, device)
        self.model_ft = copy.deepcopy(self.model_init).to(device)
        self.phase = "train"
        self.device = device
        self.dataloaders = dataloaders # get_dataloaders(dataloader_path)
        self.optimizer = get_optimizer(optim_init, self.model_ft, lr)
        self.epochs = epochs
        self.best_loss = np.inf
        self.loss_list_dict = {"train": [], "val": []}
        self.acc_list_dict = {} # ex. {"Acc": {"train": [], "val": []}, "R2": {...}}


    def step_func(self, data: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Union[float, Tuple[float, ...]]:
        """
        更新式の処理をする
        1. データを受け取る(タプルとなっているデータ等を分ける)
        2. モデルに入力し、出力を得る(モデルがtrainモードかvalモードかはmodel.trainingで取得)
        3. 損失を計算する
        4. optimizer.stepで更新する
        """
        loss_func = nn.MSELoss()
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        if self.model_ft.training:
            self.optimizer.zero_grad()
            outputs = self.model_ft(x)
            loss = loss_func(outputs.to(torch.float), y.to(torch.float))
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                outputs = self.model_ft(x)
                loss = loss_func(outputs.to(torch.float), y.to(torch.float))

        preds = outputs.detach()

        return loss.item(), (y, preds)


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
                reg_eval = eval_var[0]
                
                # r2
                targets.extend(reg_eval[0])
                preds.extend(reg_eval[1])
                
        epoch_loss = running_loss / dataset_size
        
        # r2
        targets = torch.stack(targets).cpu()
        preds = preds.stack(preds).cpu()
        epoch_acc = r2_score(targets, preds)

        return epoch_loss, {"R2": epoch_acc}
        

    def train_loop(self):
        """
        学習のループ
        """
        with tqdm(range(1, self.epochs+1)) as p_epochs:
            for epoch in p_epochs:
                p_epochs.set_description(f"[Epoch{epoch}/{self.epochs}]")

                for phase in ["train", "val"]:
                    if phase == "train":
                        self.model_ft.train()
                        self.phase = phase
                    else:
                        self.model_ft.eval()
                        self.phase = phase
                    
                    epoch_loss, eval_dict = self.epoch_func()

                    self.loss_list_dict[phase].append(epoch_loss)
                    
                    postfix = OrderedDict(Loss=epoch_loss)
                    if len(eval_dict) > 0:
                        for key in eval_dict:
                            postfix[key] = eval_dict[key]
                            if key not in self.acc_list_dict:
                                self.acc_list_dict[key] = {"train": [], "val": []}
                            else:
                                self.acc_list_dict[key][phase].append(eval_dict[key])
                    
                    p_epochs.set_postfix(
                        postfix
                    )

                    if phase == "val" and epoch_loss <= self.best_loss:
                        self.best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model_ft.state_dict())

        print("Best val Loss: {:.4f}".format(self.best_loss))
        best_idx = self.loss_list_dict["val"].index(self.best_loss)
        for key in self.acc_list_dict:
            print("{}: {:.4f}".format(key, self.acc_list_dict[key]["val"][best_idx]), end=" ")
        print()
        self.model_ft.load_state_dict(best_model_wts)
        
        
    def save(self, save_root=None, fname="model_fit"):
        save_as_pickle(self.model_ft.cpu(), fname, save_root)
        save_as_pickle(self.loss_list_dict, "loss_list_dict", save_root)
        if len(self.acc_list_dict):
            save_as_pickle(self.acc_list_dict, "acc_list_dict", save_root)

    
    def show_curve(self, logs_root: str = os.path.join(os.getcwd(), "curves/"), fname="curve", save=False):
        """
        plot loss_list_dict, acc_list_dict, cls_list_dict(which is {"train": list, "val", list})
        """
        if save:
            os.makedirs(logs_root, exist_ok=True)

        if len(self.acc_list_dict):
            _, axes = plt.subplots(1, 2, figsize=(18, 6))
            axes[0].set_title("Loss")
            axes[1].set_title("Acc")
            for phase in ["train", "val"]:
                axes[0].plot(self.loss_list_dict[phase], label=phase)
                for key in self.acc_list_dict:
                    axes[1].plot(self.acc_list_dict[key][phase], label=f"{phase} {key}")
            axes[0].legend()    
            axes[1].legend()
        else:
            plt.figure(figsize=(18, 6))
            plt.title("Loss")
            for phase in ["train", "val"]:
                plt.plot(self.loss_list_dict[phase], label=phase)
            plt.legend()
        if save:
            plt.savefig(os.path.join(logs_root, fname+".png"))
            plt.savefig(os.path.join(logs_root, fname+".pdf"))
        else:
            plt.show()
        plt.close()
