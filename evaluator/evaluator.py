import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Callable
from matplotlib.colors import TwoSlopeNorm as tsn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from ..base import BaseEvaluator
from ..utils import toColorImg


class ImageClassifier_Evaluator(BaseEvaluator):
    def __init__(self, model: nn.Module,
                 device= torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 testloaders: Dict[str, Callable[[], DataLoader]] = {},
                 logs_root: str = os.getcwd(),
                ):
        super().__init__(model=model,
                         device=device, dataloaders=dataloaders,
                         testloaders=testloaders, logs_root=logs_root)
        self.acc = 0.
        self.f_scores = {}
        self.precisions = {}
        self.recalls = {}
 
    
    def forward(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            outputs, *_ = self.model_ft(x)
            _, preds = torch.max(outputs, 1)
            
        return x, (y, preds)
    
    
    def confusion_matrix(self, folder_name:str="", fname="eval", save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
            os.makedirs(os.path.join(save_root, "reports/"), exist_ok=True)
        else:
            save_root = self.logs_root
            os.makedirs(os.path.join(save_root, "reports/"), exist_ok=True)

        labels = self.testloaders["test"].dataset.labels
        
        _, (targets, preds) = self.test(all=True)
        
        self.acc = accuracy_score(targets, preds)
        c_rep = classification_report(targets, preds, 
                                      target_names=labels,
                                      output_dict=True)

        for l in labels:
            self.f_scores[l] = c_rep[l]["f1-score"]
            self.recalls[l] = c_rep[l]["recall"]
            self.precisions[l] = c_rep[l]["precision"]
        
        plt.figure
        sns.heatmap(confusion_matrix(targets, preds), annot=True, fmt="d",
                    cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predict")
        plt.ylabel("Target")
        if save:
            plt.savefig(os.path.join(save_root, fname+".png"))
            plt.savefig(os.path.join(save_root, fname+".pdf"))
            with open(os.path.join(save_root, "reports/", f"{fname}.txt"), "w") as f:
                print(classification_report(targets, preds, target_names = labels), file=f)
        else:
            print(classification_report(targets, preds, target_names = labels))
            plt.show()
        plt.close()
        
        
    # def evaluate(self, threshold=0.1):
    #     """
    #     クラス分類での評価であるAccuracyと誤答率、不明率を算出する。
    #     不明率は、ソフトマックスにおける数値のトップｋが平坦であれば不明だとする。
    #     """
    #     labels = self.testloaders["test"].dataset.labels
    #     data_size = len(self.testloaders["test"].dataset)
    #     match = 0.
    #     wrongs = 0.
    #     unknown = 0.

    #     for data in self.testloaders["test"]:
    #         x, y = data
    #         x, y = x.to(self.device), y.to(self.device)
        
    #         with torch.no_grad():
    #             outputs, *_ = self.model_ft(x)
    #             probabilities = torch.nn.functional.softmax(outputs.squeeze(), dim=0)

    #             top_prob, top_catid = torch.topk(probabilities, len(labels))
                
    #             if torch.count_nonzero(top_prob[0] - top_prob[1] < threshold) > 0:
    #                 unknown += torch.count_nonzero(top_prob[0] - top_prob[1] < threshold).item()
    #             else:
    #                 match += torch.count_nonzero(top_catid[0] == y).item()
    #                 wrongs += torch.count_nonzero(top_catid[0] != y).item()
                
                
    #     return match/data_size, wrongs/data_size, unknown/data_size
        
        
    def show_scores(self, folder_name:str="", fname="eval", save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
        else:
            save_root = self.logs_root

        labels = self.testloaders["test"].dataset.labels
        acc_format = f"Accuracy: {self.acc:.4f}"
        fscore_format = "F scores: "
        recall_format = "Recalls: "
        precision_format = "Precisions: "
        for l in labels:
            fscore_format += "({}: {:.4f})".format(l, self.f_scores[l])
            recall_format += "({}: {:.4f})".format(l, self.recalls[l])
            precision_format += "({}: {:.4f})".format(l, self.precisions[l])
        
        if save:
            with open(save_root+f"/{fname}.txt", "w") as f:
                print(acc_format, file=f)
                print(fscore_format, file=f)
                print(recall_format, file=f)
                print(precision_format, file=f)
        else:
            print(acc_format)
            print(fscore_format)
            print(recall_format)
            print(precision_format)
            
            
    def show_cam(self, folder_name:str="", fname="cam", random=True, save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
        else:
            save_root = self.logs_root
        
        is_random = ["test", "randomtest"][int(random)]
        labels = self.testloaders["test"].dataset.labels
        
        imgs, targets = next(iter(self.testloaders[is_random]))
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs, fmaps = self.model_ft(imgs)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(outputs.squeeze(), dim=0)

        # Show the classification result
        top_prob, top_catid = torch.topk(probabilities, len(labels))
        print("Classified as \033[1m", labels[top_catid[0]], "\033[0m, model output =", outputs.squeeze())
        for i in range(top_prob.size(0)):
            print(labels[top_catid[i]], "(", top_catid[i].cpu().numpy(), ")", "{:.2f} %".format(top_prob[i].item()*100))
            
        weight = self.model_ft.classifier[0].weight
            
        class_activation_mapper = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1, padding=0, bias=False)
        class_activation_mapper.weight = torch.nn.Parameter(weight.unsqueeze(-1).unsqueeze(-1))
        upsample = nn.Upsample(tuple(imgs.shape[-2:]), mode="bilinear")
            
        with torch.no_grad():
            cam = class_activation_mapper(fmaps)
            cam_avgs = cam.mean([-2,-1]).cpu().numpy().tolist()
        
        if imgs.shape[1] != 3:
            imgs = imgs.repeat(1, 3, 1, 1)
        in_img = imgs.squeeze().cpu().permute(1, 2, 0).numpy()
        cam = upsample(cam).squeeze().cpu().numpy()
        ccam = toColorImg(cam[top_catid[0]], cm='bwr')#.transpose(2,0,1)
        
        # overlay cam on input
        alpha = 0.25
        cam_image = in_img*alpha + ccam*(1-alpha)
        # print(cam_image.min(), cam_image.max())
        
        fig, ax = plt.subplots(2, 1, figsize=(30, 5))
        fig.suptitle(f"Target: {labels[targets.item()]}, Prediction: {labels[top_catid[0]]}")
        # ax[0].imshow(in_img, vmin=0, vmax=1)
        ax[0].imshow(in_img)
        ax[1].imshow(cam_image, vmin=0, vmax=1)
        if save:
            plt.savefig(os.path.join(save_root, fname+".png"))
            plt.savefig(os.path.join(save_root, fname+".pdf"))
        else:
            plt.show()
        plt.close()
        

        
class ABN_Evaluator(ImageClassifier_Evaluator):
    def __init__(self, model: nn.Module,
                 device= torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 testloaders: Dict[str, Callable[[], DataLoader]] = {},
                 logs_root: str = os.getcwd(),
                ):
        super().__init__(model=model,
                         device=device, dataloaders=dataloaders,
                         testloaders=testloaders, logs_root=logs_root)
    
    def forward(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            outputs, *_ = self.model_ft(x)
            _, preds = torch.max(outputs, 1)
            
        return x, (y, preds)
            
            
    def show_cam(self, folder_name:str="", fname="cam", random=True, save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
        else:
            save_root = self.logs_root
        
        is_random = ["test", "randomtest"][int(random)]
        labels = self.testloaders["test"].dataset.labels
        
        imgs, targets = next(iter(self.testloaders[is_random]))
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs, att_out, fmaps, attention_map = self.model_ft(imgs)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(outputs.squeeze(), dim=0)

        # Show the classification result
        top_prob, top_catid = torch.topk(probabilities, len(labels))
        print("Classified as \033[1m", labels[top_catid[0]], "\033[0m, model output =", outputs.squeeze())
        for i in range(top_prob.size(0)):
            print(labels[top_catid[i]], "(", top_catid[i].cpu().numpy(), ")", "{:.2f} %".format(top_prob[i].item()*100))
            
        weight = self.model_ft.classifier[0].weight
            
        class_activation_mapper = torch.nn.Conv2d(weight.shape[1], weight.shape[0], 1, padding=0, bias=False)
        class_activation_mapper.weight = torch.nn.Parameter(weight.unsqueeze(-1).unsqueeze(-1))
        upsample = nn.Upsample(tuple(imgs.shape[-2:]), mode="bilinear")
            
        with torch.no_grad():
            cam = class_activation_mapper(fmaps)
            cam_avgs = cam.mean([-2,-1]).cpu().numpy().tolist()
        
        if imgs.shape[1] != 3:
            imgs = imgs.repeat(1, 3, 1, 1)
        in_img = imgs.squeeze().cpu().permute(1, 2, 0).numpy()
        cam = upsample(cam).squeeze().cpu().numpy()
        ccam = toColorImg(cam[top_catid[0]], cm='bwr')#.transpose(2,0,1)
        
        # overlay cam on input
        alpha = 0.25
        cam_image = in_img*alpha + ccam*(1-alpha)
        # print(cam_image.min(), cam_image.max())
        
        fig, ax = plt.subplots(2, 1, figsize=(30, 5))
        fig.suptitle(f"Target: {labels[targets.item()]}, Prediction: {labels[top_catid[0]]}")
        # ax[0].imshow(in_img, vmin=0, vmax=1)
        ax[0].imshow(in_img)
        ax[1].imshow(cam_image, vmin=0, vmax=1)
        if save:
            plt.savefig(os.path.join(save_root, fname+".png"))
            plt.savefig(os.path.join(save_root, fname+".pdf"))
        else:
            plt.show()
        plt.close()
        
        
    def show_attention(self, folder_name:str="", fname="attention_map", random=True, save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
        else:
            save_root = self.logs_root
        
        is_random = ["test", "randomtest"][int(random)]
        labels = self.testloaders["test"].dataset.labels
        
        imgs, targets = next(iter(self.testloaders[is_random]))
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs, att_out, fmaps, attention_map = self.model_ft(imgs)
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(outputs.squeeze(), dim=0)

        # Show the classification result
        top_prob, top_catid = torch.topk(probabilities, len(labels))
        print("Classified as \033[1m", labels[top_catid[0]], "\033[0m, model output =", outputs.squeeze())
        for i in range(top_prob.size(0)):
            print(labels[top_catid[i]], "(", top_catid[i].cpu().numpy(), ")", "{:.2f} %".format(top_prob[i].item()*100))
            
        upsample = nn.Upsample(tuple(imgs.shape[-2:]), mode="bilinear")
        
        if imgs.shape[1] != 3:
            imgs = imgs.repeat(1, 3, 1, 1)
        in_img = imgs.squeeze().cpu().permute(1, 2, 0).numpy()
        attention_mask = upsample(attention_map).squeeze().cpu().numpy()
        attention_mask = toColorImg(attention_mask, cm='bwr')#.transpose(2,0,1)
        
        # overlay cam on input
        alpha = 0.25
        attention_image = in_img*alpha + attention_mask*(1-alpha)
        # print(cam_image.min(), cam_image.max())
        
        fig, ax = plt.subplots(3, 1, figsize=(18, 5))
        fig.suptitle(f"Target: {labels[targets.item()]}, Prediction: {labels[top_catid[0]]}")
        # ax[0].imshow(in_img, vmin=0, vmax=1)
        ax[0].imshow(in_img)
        ax[1].imshow(attention_map.squeeze().cpu().numpy(), cmap="hot")
        ax[2].imshow(attention_image, vmin=0, vmax=1)
        if save:
            plt.savefig(os.path.join(save_root, fname+".png"))
            plt.savefig(os.path.join(save_root, fname+".pdf"))
        else:
            plt.show()
        plt.close()
        
        
class LSUnet_Evaluator(BaseEvaluator):
    def __init__(self, model: nn.Module,
                 device= torch.device("cpu"),
                 dataloaders: Dict[str, Callable[[], DataLoader]] = {},
                 testloaders: Dict[str, Callable[[], DataLoader]] = {},
                 logs_root: str = os.getcwd(),
                ):
        super().__init__(model=model,
                         device=device, dataloaders=dataloaders,
                         testloaders=testloaders, logs_root=logs_root)
        
        
    def show_lowrank_sparse(self, folder_name:str="", fname="lowrank_sparse", random=True, save=False):
        if len(folder_name):
            save_root = os.path.join(self.logs_root, folder_name)
            os.makedirs(save_root, exist_ok=True)
        else:
            save_root = self.logs_root
        
        is_random = ["test", "randomtest"][int(random)]
        labels = self.testloaders["test"].dataset.labels
        
        imgs, targets = next(iter(self.testloaders[is_random]))
        imgs = imgs.to(self.device)
        with torch.no_grad():
            out_s, fmaps = self.model_ft(imgs)
            l = imgs - out_s
        
        fig, ax = plt.subplots(3, 1, figsize=(30, 6))
        ax[0].imshow(imgs[0].cpu().squeeze().numpy(), cmap="gray")
        ax[1].imshow(out_s[0].cpu().squeeze().numpy(), cmap="gray")
        ax[2].imshow(l[0].cpu().squeeze().numpy(), cmap="gray")
        if save:
            plt.savefig(os.path.join(save_root, fname+".png"))
            plt.savefig(os.path.join(save_root, fname+".pdf"))
        else:
            plt.show()


    def plotFeaturemaps(self, ncols=4, nrows=8, figsize=None, mag=1, random=True):
        is_random = ["test", "randomtest"][int(random)]
        labels = self.testloaders["test"].dataset.labels
        
        imgs, targets = next(iter(self.testloaders[is_random]))
        imgs = imgs.to(self.device)
        with torch.no_grad():
            out_s, fmaps = self.model_ft(imgs)
        maps = fmaps.cpu().squeeze().numpy()
        
        nmaps = nrows * ncols
        channels = range(nmaps)

        norm = tsn(vmin=np.minimum(maps[:].min(),-1e-6), vcenter=0, vmax=np.maximum(maps[:].max(),1e-6)) # maps[channels]
        fig = plt.figure(figsize=figsize if figsize is not None else (ncols*mag, nrows*mag))
        for ch in range(nmaps):
            ax1 = fig.add_subplot(nrows, ncols, ch+1)
            ax1.imshow(maps[channels[ch]], cmap='bwr_r', norm=norm)#, aspect='equal')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        plt.tight_layout()
        plt.show()

        