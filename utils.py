import os
from .pickle_io import save_as_pickle, read_as_pickle
from torch.optim import Optimizer
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
from matplotlib import cm as cmap
from matplotlib.colors import TwoSlopeNorm as tsn
from PIL.Image import Image

# x[H,W]
def toColorImg(x, cm="bwr", norm=None):
    if norm is not None:
        norm = tsn(vmin=np.minimum(x[:].min(),-1e-6), vcenter=0, vmax=np.maximum(x[:].max(),1e-6)) # maps[channels]
    sm = cmap.ScalarMappable(norm=norm, cmap=cm)
    return sm.to_rgba(x)[:,:,:3]

def reset_weight(model: nn.Module):
    """
    Try resetting model weights to avoid weight leakage.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()
    return model


def get_model(fname: str, model_root: str, device=torch.device("cpu")):
    try:
        model = read_as_pickle(fname, model_root)
        model.to(device)
        return model
    except Exception as e:
        print(e)

def set_model(model, fname: str="nontrain", model_root: str=os.getcwd()):
    os.makedirs(model_root, exist_ok=True)
    save_as_pickle(model.cpu(), fname, model_root)

def PILToTensor(x: Image):
    return torch.from_numpy( np.array(x).astype(np.float32) ).clone()
