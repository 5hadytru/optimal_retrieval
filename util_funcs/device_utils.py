import torch
import transformers

def get_device(id=None):
    if not id is None and not id == -1:
        return torch.device(f"cuda:{id}")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    try: # for HuggingFace
        data = data["pixel_values"]
        if not len(data) == 1:
            print("len(data['pixel_values'] was not 1; must be in the online regime")
            exit(0)
        data = data[0]
    except:
        pass
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device=device)


class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.data)
