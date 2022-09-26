from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.x = data[0]
        self.y = data[1]
        self.transform = transform
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        x = self.x[idx].detach().clone()
        y = self.y[idx].detach().clone()
        
        if self.transform is not None:
            x = self.transform(x)
            
        return x, y
