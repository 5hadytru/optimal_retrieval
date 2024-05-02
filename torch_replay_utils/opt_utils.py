import torch
import time
import wandb
from abc import ABC, abstractmethod
from importlib.machinery import SourceFileLoader
import torch.distributed as dist

# import custom modules
models = SourceFileLoader("models", "models/models.py").load_module()

concat_outputs = lambda fd1, y1, fd2, y2: (
    {k: torch.cat([fd1[k], fd2[k]], dim=0) for k in fd1.keys()}, 
    {k: torch.cat([y1[k], y2[k]], dim=0) for k in y1.keys()}
)

concat_batch = lambda i1, i2, y1, y2: (
    {k: torch.cat([i1[k], i2[k]], dim=0) for k in i1.keys() if k in i1 and k in i2},
    {k: torch.cat([y1[k], y2[k]], dim=0) for k in y1.keys() if k in y1 and k in y2},
)

def split_batch(inputs:dict, fd:dict, y:dict):
    batch_indices = torch.nonzero(~inputs["ds_id"]).squeeze()
    replay_indices = torch.nonzero(inputs["ds_id"]).squeeze()

    batch_fd = {k: v[batch_indices] for k, v in fd.items()}
    batch_y = {k: v[batch_indices] for k, v in y.items() if "denom" not in k}

    replay_fd = {k: v[replay_indices] for k, v in fd.items()}
    replay_y = {k: v[replay_indices] for k, v in y.items() if "denom" not in k}

    if batch_indices.numel() == 1:
        batch_fd = {k: v.unsqueeze(0) for k, v in batch_fd.items()}
        batch_y = {k: v.unsqueeze(0) for k, v in batch_y.items()}

    if replay_indices.numel() == 1:
        replay_fd = {k: v.unsqueeze(0) for k, v in replay_fd.items()}
        replay_y = {k: v.unsqueeze(0) for k, v in replay_y.items()}

    batch_y = batch_y | {k: v[0] for k, v in y.items() if "denom" in k}
    replay_y = replay_y | {k: v[1] for k, v in y.items() if "denom" in k}

    return batch_fd, batch_y, replay_fd, replay_y, replay_indices

log_rank_0 = lambda d: wandb.log(d) if dist.get_rank() == 0 else None
print_rank_0 = lambda d: print(d) if dist.get_rank() == 0 else None


def get_cls_denom(labels:torch.Tensor):
    if labels.numel() == 0:
        cls_denom = torch.tensor([0.]).to(torch.device(dist.get_rank()))
        contributed = torch.tensor(0.).to(torch.device(dist.get_rank()))
    else:
        cls_denom = torch.tensor([labels[...,1:].sum()]).to(torch.float).to(torch.device(dist.get_rank()))
        contributed = torch.tensor(1.).to(torch.device(dist.get_rank()))

    dist.all_reduce(cls_denom)
    dist.all_reduce(contributed)
    cls_denom /= contributed
    cls_denom = torch.maximum(cls_denom, torch.tensor([1.]).to(torch.device(dist.get_rank())))

    return cls_denom
    

def get_box_denom(boxes:torch.Tensor, labels:torch.Tensor):
    if boxes.numel() == 0:
        box_denom = torch.tensor([0.]).to(torch.device(dist.get_rank()))
        contributed = torch.tensor(0.).to(torch.device(dist.get_rank()))
    else:
        n_labels_per_instance = torch.sum(labels[..., 1:], dim=-1)
        box_denom = torch.tensor([torch.sum(n_labels_per_instance > 0, dtype=torch.float)]).to(torch.device(dist.get_rank()))
        contributed = torch.tensor(1.).to(torch.device(dist.get_rank()))

    dist.all_reduce(box_denom)
    dist.all_reduce(contributed)
    box_denom /= contributed
    box_denom = torch.maximum(box_denom, torch.tensor([1.]).to(torch.device(dist.get_rank())))

    return box_denom


def add_denoms(y:dict):
    """
    Given local boxes and labels, add shared (across devices) denominators for box and label loss normalization
    The shared value of cls_denom is the average number of positive labels (1s) per-image across devices
    The shared value of box_denom is the average number of boxes per-image across devices
    """
    cls_denom = get_cls_denom(y["labels"])
    box_denom = get_box_denom(y["boxes"], y["labels"])

    y["cls_denom"] = cls_denom
    y["box_denom"] = box_denom


"""
ABCs for executing a forward pass
"""
class BatchForwardPass(ABC):
    """
    Base class for batch forward passes, which take in a batch in the standard 'X,y = batch' format
    """
    def __init__(self, cfg):
        self.cfg = cfg
    
    @abstractmethod
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        pass

"""
Implementations which assume that all data is present in (X,y)
"""
class Basic(BatchForwardPass):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        """
        Basic forward & loss computation; can execute basic SGD and ER
        """
        forward_dict = model(**inputs)
        add_denoms(y)
        total_mean_loss, total_loss_each, metrics = model.module.loss_func(forward_dict | y)

        log_rank_0({"loss": total_mean_loss.detach()})
    
        return total_mean_loss
    

class NoGrad(BatchForwardPass):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        with torch.no_grad():
            """
            Assumes (X, y) contains all data  to compute loss/gradients/features with
            """
            forward_dict = model(**inputs)
            add_denoms(y)
            total_mean_loss, total_loss_each, metrics = model.module.loss_func(forward_dict | y)
            
            return total_loss_each.detach() # should already be detached


class DER(BatchForwardPass):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        """
        Use this for mixed batches (replay and new samples)
        """
        forward_dict = model(**inputs)

        batch_fd, batch_y, replay_fd, replay_y, replay_indices = split_batch(inputs, forward_dict, y)

        add_denoms(batch_y)
        if batch_y["labels"].numel() > 0:
            batch_mean_loss, _, __ = model.module.loss_func(batch_fd | batch_y)
            log_rank_0({'batch_loss': batch_mean_loss.item()})
        else:
            batch_mean_loss = torch.tensor(0)

        add_denoms(replay_y)
        if replay_fd["pred_logits"].numel() > 0:
            der_loss = self.compute_DER_loss(model.module.loss_func, replay_fd, replay_y, inputs["logits"][replay_indices])
            if batch_mean_loss.item() == 0:
                return der_loss, replay_indices.numel()
            else:
                return batch_mean_loss + der_loss, replay_indices.numel()
        else:
            return batch_mean_loss, replay_indices.numel()
        

    def compute_mse(self, x, y):
        if len(y.shape) == 2:
            return torch.mean(torch.mean((x - y) ** 2, dim=2), dim=1)
        else:
            # need to avg across the batch
            return torch.mean(torch.mean(torch.mean((x - y) ** 2, dim=2), dim=1))
    
    
    def compute_DER_loss(self, loss_func, replay_fd:dict, replay_y:dict, old_logits:torch.Tensor):
        cls_loss, _, __ = loss_func(replay_fd | replay_y)
        logit_loss = self.compute_mse(replay_fd["pred_logits"], old_logits)

        scaled_logit_loss = self.cfg.main.DER_alpha * logit_loss
        scaled_cls_loss = self.cfg.main.DER_beta * cls_loss

        log_rank_0({
            "ps_der_logit_loss": logit_loss.item(), 
            "ps_der_cls_loss": cls_loss.item(),
            "der_logit_loss": scaled_logit_loss.item(), 
            "der_cls_loss": scaled_cls_loss.item()
        })

        return scaled_logit_loss + scaled_cls_loss


class Random(BatchForwardPass):
    """
    Forward pass with random retrieval
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.der_cls = DER(cfg)
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        replay_batch, replay_y = kwargs["retrieve_func"](inputs["images"].size(0))
        combined_inputs, combined_y = concat_batch(inputs, replay_batch, y, replay_y)

        forward_dict = model(**combined_inputs)

        batch_fd, batch_y, replay_fd, replay_y, replay_indices = split_batch(combined_inputs, forward_dict, combined_y)
        add_denoms(batch_y)

        batch_loss_mean, _, __ = model.module.loss_func(batch_fd | batch_y)
        log_rank_0({'batch_loss': batch_loss_mean.item()})

        if replay_batch["logits"].numel() > 0:
            add_denoms(replay_y)
            return batch_loss_mean + self.der_cls.compute_DER_loss(
                model.module.loss_func, replay_fd, replay_y, combined_inputs["logits"][replay_indices])
        else:
            return batch_loss_mean


class Features(BatchForwardPass):
    """
    Forward pass with feature-based retrieval
    """
    def __init__(self, cfg, cls_embeds:bool):
        super().__init__(cfg)
        self.der_cls = DER(cfg)
        self.cls_embeds = cls_embeds
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        """
        Had to do a separate no_grad forward pass since the HF OWL-ViT implementation
        is incompatible with grad accumulation (can't do multiple, separate forward passes for one
        grad step)

        Note that this is not slower than just calling forward()
        """
        with torch.no_grad():
            query_embeds = model.module.get_text_embeddings(
                inputs["input_ids"].view(-1, inputs["input_ids"].size(-1)), 
                inputs["attention_mask"].view(-1, inputs["input_ids"].size(-1))
            )
            query_embeds = query_embeds.view(inputs["input_ids"].size(0), inputs["input_ids"].size(1), query_embeds.size(-1))
            image_features = model.module.get_image_features(inputs["images"], return_cls_token=False) # (B, num ViT tokens, ViT token dim)
            pred_boxes = model.module.get_boxes(image_features)
            pred_logits, cls_embeds = model.module.get_logits(image_features, query_embeds, return_embeds=True) # (B, num ViT tokens, sum(num prompts per class))
            add_denoms(y)
            _, loss_each, __ = model.module.loss_func({"pred_logits": pred_logits, "pred_boxes": pred_boxes} | y)

        replay_batch, replay_y = kwargs["retrieve_func"](cls_embeds if self.cls_embeds else image_features, query_embeds, pred_logits)

        combined_inputs, combined_y = concat_batch(inputs, replay_batch, y, replay_y)

        forward_dict = model(**combined_inputs)

        batch_fd, batch_y, replay_fd, replay_y, replay_indices = split_batch(combined_inputs, forward_dict, combined_y)

        add_denoms(batch_y)
        batch_loss_mean, _, __ = model.module.loss_func(batch_fd | batch_y)
        log_rank_0({'batch_loss': batch_loss_mean.item()})

        if self.cfg.main.loss_adaptive and batch_loss_mean.item() < self.cfg.main.replay_thresh:
            # NOTE: this implementation does not prevent calling forward() on unnecessary replay batches
            # add_denoms requires an all_reduce, so we must fill-in the box and label tens with empty tensors and call add_denoms 
            print_rank_0("Used loss adaptivity")
            add_denoms({
                "boxes": torch.tensor([], dtype=torch.float32, device=torch.device(dist.get_rank())),
                "labels": torch.tensor([], dtype=torch.long, device=torch.device(dist.get_rank()))
            })
            return batch_loss_mean
        else:
            print_rank_0("Using replay")

        if replay_batch["logits"].numel() > 0:
            add_denoms(replay_y)
            return batch_loss_mean + self.der_cls.compute_DER_loss(
                model.module.loss_func, replay_fd, replay_y, combined_inputs["logits"][replay_indices])
        else:
            return batch_loss_mean
        

"""
class ANNR(BatchForwardPass):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def __call__(self, inputs:dict, y:dict, model, **kwargs):
        outputs = model(**inputs, feature_type=self.cfg.retrieval.annr.feature_type)

        # get batch items' losses for *adaptive* NNR
        with torch.no_grad():
            total_mean_loss, total_loss_each, metrics = model.module.loss_func(outputs, y)

        # get nearest neighbor replay batches
        start = time.time()
        knn_batch = kwargs["retrieve_func"](batch_features=outputs["features"], pre_SGD_losses=total_loss_each)
        log_rank_0({"retrieve_s": time.time() - start})

        if knn_batch["X"].numel() > 0:
            knn_forward_dict = model(knn_batch["X"], knn_batch["y"], feature_type=self.cfg.retrieval.annr.feature_type)

            # make it as if the two model() calls were a single call
            combined_forward_dict = combine_forward_dicts(self.cfg.main.model_opt, outputs, knn_forward_dict)

            # compute loss
        else:
            total_mean_loss, total_loss_each, metrics = model.module.loss_func(
                outputs, 
                y
            )
        
        return total_mean_loss
"""
