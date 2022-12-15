import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractclassmethod


class GradRAM_Model(nn.Module):
    @abstractclassmethod
    def get_features(self, x):
        pass
    
    @abstractclassmethod
    def get_outputs(self, x):
        pass
    
    def forward(self, x):
        x = self.get_features(x)
        return self.get_outputs(x)

    def get_fmaps_alpha(self, x):
        fmaps = self.get_features(x)
        fmaps = fmaps.clone().detach().requires_grad_(True)
        outputs = self.get_outputs(fmaps)
        outputs.backward()
        alpha = fmaps.grad.mean([-2, -1])
        
        return fmaps, alpha
    
    def get_cam(self, x):
        fmaps, alpha = self.get_fmaps_alpha(x)
        feature_num, cls_num = alpha.shape[1], alpha.shape[0]
        cam_mapper = nn.Conv2d(feature_num, cls_num, kernel_size=1, padding=0, bias=False)
        cam_mapper.weight = nn.Parameter(alpha.unsqueeze(-1).unsqueeze(-1))
        
        with torch.no_grad():
            cam = cam_mapper(fmaps)
        
        return cam

class GradCAM_Model(nn.Module):
    @abstractclassmethod
    def get_features(self, x):
        pass
    
    @abstractclassmethod
    def get_outputs(self, x):
        pass
    
    def forward(self, x):
        x = self.get_features(x)
        return self.get_outputs(x)

    def get_fmaps_alpha(self, x):
        fmaps = self.get_features(x)
        fmaps = fmaps.clone().detach().requires_grad_(True)
        outputs = self.get_outputs(fmaps)
        pred_idx = torch.argmax(outputs)
        outputs[:, pred_idx].backward()
        alpha = fmaps.grad.mean([-2, -1])
        
        return fmaps, alpha

    def get_cam(self, x):
        fmaps, alpha = self.get_fmaps_alpha(x)
        feature_num, cls_num = alpha.shape[1], alpha.shape[0]
        cam_mapper = nn.Conv2d(feature_num, cls_num, kernel_size=1, padding=0, bias=False)
        cam_mapper.weight = nn.Parameter(alpha.unsqueeze(-1).unsqueeze(-1))
        
        with torch.no_grad():
            cam = cam_mapper(fmaps)
            cam = F.relu(cam)
        
        return cam
