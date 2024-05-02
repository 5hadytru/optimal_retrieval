from torchsummary import summary
import torch.nn as nn
import torch
import sys, os
from importlib.machinery import SourceFileLoader
from PIL import Image
from transformers import OwlViTForObjectDetection

# import custom modules
if __name__ == "__main__":
    models = SourceFileLoader("models", "../models/models.py").load_module()
else:
    models = SourceFileLoader("models", "models/models.py").load_module()

tower_indices = {
    "OWL_B16": [197],
    "OWL_L14": [197]
}

model_index = {
    "OWL_B16": models.OWL_B16,
    "OWL_L14": models.OWL_L14
}

init_model = lambda model_name: model_index[model_name]()

def apply_lw_lr_decay(model, model_name:str, lrs:list, lr_mults:list):
    """
    Get the optimizer inputs (list of dicts) to apply layerwise learning rate decay

    Args:
        lrs: list of maximum learning rates. First LR is for the last tower in model.named_parameters(), last is for first. This is because we .reverse() the list of names below
        lr_mults: list of LR coefficients to apply at each layer
    Returns:
        list[dict]
    """
    # get descending list of indices indicating first indices for each tower after the first
    next_tower_idxs = tower_indices[model_name]

    # get ordered list of parameter names
    layer_names = []
    for idx, (name, _) in enumerate(model.named_parameters()):
        layer_names.append((idx, name))

    layer_names.reverse() # now starts at the final layer of tower 2 (or tower 1)

    parameters = []

    # store params & learning rates
    current_tower = 0
    current_lr = lrs[current_tower]
    for idx, name in layer_names:
        # if this is a multi-tower model (ex: two towers of CLIP), restart the process for the next model
        if len(next_tower_idxs) > current_tower and idx == next_tower_idxs[current_tower]:
            current_tower += 1
            current_lr = lrs[current_tower]
        
        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':     current_lr}]
        
        # update learning rate
        current_lr *= lr_mults[current_tower]

    return parameters


def get_model_info(model_name, device):
    model = model_index[model_name](device).to(device)
    texts = [["a photo of a cat", "a photo of a dog"]]
    inputs = model.processor(text=texts, images=Image.open("../dog.png"), return_tensors="pt")

    out_dict = model(inputs["pixel_values"].to(device), inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

    print(out_dict["text_model_last_hidden_state"].size(), out_dict["vision_model_last_hidden_state"].size())

    for idx, (name, _) in enumerate(model.named_parameters()):
        print((idx, name))


def write_layers_to_file(model, fname1, input_shape):
    with open(fname1, "w") as f:
        f.write(str([module for module in model.modules() if not isinstance(module, nn.Sequential)]))
    summary(model, input_shape)


if __name__ == "__main__":
    model_name, model = "OWL_L14", OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
    apply_lw_lr_decay(model, model_name, [0.0001, 0.001], [0.9,0.9])