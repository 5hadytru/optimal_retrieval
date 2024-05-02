from re import L
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import wandb
import os, sys
from importlib.machinery import SourceFileLoader
import omegaconf
import numpy as np
import gc, time, math
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from lvis.eval import LVISEval
from lvis.lvis import LVIS
from lvis.results import LVISResults
import functools
from torch.distributed import init_process_group
import torch.distributed as dist
import pickle, socket


COCO_METRIC_NAMES = [
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
]


def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def wandb_log_rank_0(log_dict:dict):
    if dist.get_rank() == 0:
        wandb.log(log_dict)


def validate_cfg(cfg):
    if (cfg.main.chkpt_path is not None) and (not cfg.main.test_only):
        raise Exception("Only able to load checkpoints if testing only")

    if cfg.main.testing not in [0,1]:
        raise Exception("Must provide cfg.main.testing in [0,1]")

    if cfg.main.wandb_proj is None and cfg.main.wandb_mode == "online":
        raise Exception("Must specify wandb_project if wandb_mode is online")
    
    assert cfg.main.reset_dedup_on in ["dataset", "epoch", "proportion", "batch"]
    assert not (cfg.main.reset_dedup_on == "proportion" and cfg.main.dedup_prop is None)


def free_all_datasets(loader_dict):
    """
    First, delete loaders referring to the datasets
    """
    for split, split_dict in loader_dict.items():
        if split == "label_to_captions":
            continue

        if split == "train":
            for ds_dict in split_dict.values():
                dels = []
                for key, val in ds_dict.items():
                    if key == "loader":
                        dels.append(key)
                for key in dels:
                    del ds_dict[key].data # ToDeviceLoader is a wrapper
                    del ds_dict[key]
        else:
            dels = []
            for key, val in split_dict.items():
                dels.append(key)
            for key in dels:
                del split_dict[key].data # ToDeviceLoader is a wrapper
                del split_dict[key]

    gc.collect()

    """
    Then, free the memory and collect all garbage
    """
    objects = gc.get_objects()

    # iterate over the objects and delete TensorDataset objects that are above the max size
    for obj in objects:
        if isinstance(obj, torch.utils.data.TensorDataset):
            #print_rank_0("--", len(gc.get_referrers(obj)))
            for ref in gc.get_referrers(obj):
                #print_rank_0("----", len(gc.get_referents(ref)))
                for referent in gc.get_referents(ref):
                    del referent
                del ref
                gc.collect()

            # size = obj.tensors[0].element_size() * obj.tensors[0].nelement()
            #print_rank_0("-------gc", obj, obj.tensors[0].element_size() * obj.tensors[0].nelement() / 1000000000)
            del obj
            gc.collect()


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_batch(batch, prompt_type:str, device, images_only:bool=False):
    """
    Return inputs and targets such that they're ready to be passed to the trainer
    """
    if len(batch) == 4:
        return batch[0].to(device), batch[1].to(device) # some datasets (e.g LVIS) only have tensorized images since they are never trained on
    else:
        images, image_ids, imagenet_input_ids, imagenet_attention_mask, custom_input_ids, custom_attention_mask, boxes, labels, logits, ds_id = batch

    if images_only:
        return images.to(device), image_ids.to(device)

    inputs = {"images": images, "logits": logits, "ds_id": ds_id}
    y = {"boxes": boxes, "labels": labels.to(torch.int)}

    # can use imagenet or custom prompt templates
    if prompt_type == "custom":
        inputs["input_ids"] = custom_input_ids
        inputs["attention_mask"] = custom_attention_mask
    elif prompt_type == "imagenet":
        inputs["input_ids"] = imagenet_input_ids
        inputs["attention_mask"] = imagenet_attention_mask
    else:
        raise Exception(prompt_type)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    for k, v in y.items():
        y[k] = v.to(device)

    return inputs, y


def unravel_index_2D(indices:torch.Tensor, shape:torch.Size):
    """
    Torch implementation of numpy unravel_index for two dimensions
    """
    assert len(shape) == 2 and len(indices.shape) == 1

    shape = torch.tensor(shape)
    rows = indices // shape[1]
    cols = indices % shape[1]

    return (rows, cols)


@functools.partial(torch.vmap, in_dims=(0, 0, None, None))
def get_top_k_preds(logits:torch.Tensor, boxes:torch.Tensor, k:int, exclusive_classes:bool):
    """
    Finds the top k scores and corresponding boxes within an image.

    The code applies on the image level; must use vmap for batching.
    
    Args:
    scores: [num_instances, num_classes] array of scores (i.e. logits or
      probabilities) to sort by.
    boxes: [num_instances, 4] Optional array of bounding boxes.
    k: Number of instances to return.
    exclusive_classes: If True, the top class for each box is returned. If
      False, classes are considered to be non-exclusive (multi-label setting),
      and the top-k computations happens globally across all scores, not just
      the maximum logit for each output token.

    Returns:
    Score, label, and box arrays of shape [top_k, ...] for the selected
    instances.
    """
    assert len(logits.shape) == 2

    if exclusive_classes:
        k = min(k, logits.shape[0]) # k cannot be greater than the number of ViT tokens
        instance_top_scores, instance_class_ind = torch.max(logits, dim=1) # (num ViT tokens), (num ViT tokens)
        top_scores, instance_ind = torch.topk(instance_top_scores, k) # (k), (k)
        class_ind = instance_class_ind[instance_ind] # (k)
    else:
        k = min(k, logits.numel()) # k cannot be greater than the total number of class predictions
        top_scores, top_indices = torch.topk(logits.view(-1), k) # (k), (k)
        instance_ind, class_ind = unravel_index_2D(top_indices, logits.shape) # (k), (k)

    return top_scores, class_ind, boxes[instance_ind] # (k, k, (k,4))


def get_test_queries(prompts:dict, query_emb_dim:float, processor, model):
    """
    Get test queries for each class for each of the best 7 prompt templates
    """
    sorted_class_names = sorted(prompts.keys())
    assert len(sorted_class_names) == len(set(sorted_class_names)) # ensure no duplicate class names

    num_prompts_for_each_class = [len(prompts[class_name]) for class_name in sorted_class_names]

    preprocessed_input = processor(
        text=[[prompt for prompt in prompts[class_name]] for class_name in sorted_class_names],
        return_tensors='pt',
        truncation=False
    )

    embeddings = model.module.get_text_embeddings(
        preprocessed_input["input_ids"].to(torch.device(dist.get_rank())),
        preprocessed_input["attention_mask"].to(torch.device(dist.get_rank()))
    )

    assert list(embeddings.shape) == [sum(num_prompts_for_each_class), query_emb_dim], str(embeddings.shape)

    return embeddings, num_prompts_for_each_class, sorted_class_names


def format_predictions(predictions:dict, dsi_split_dict:dict, k_preds_per_image:int):
    """
    Format all predictions in COCO format: [{image_id:int, category_id: int, bbox: [x, y, w, h], score: float}]
    """
    # (cx, cy, w, h) all in range [0,1] -> [xmin, ymin, w, h]
    img_ws, img_hs = [dsi_split_dict[int(img_id)]["width"] for img_id in predictions["image_id"]], [dsi_split_dict[int(img_id)]["height"] for img_id in predictions["image_id"]]
    boxes = predictions["bbox"]
    boxes *= torch.tensor([img_ws, img_hs, img_ws, img_hs]).transpose(0,1).unsqueeze(1).to(boxes.device)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2.0

    formatted_predictions = []
    for i in range(predictions["image_id"].size(0)):
        img_id = int(predictions["image_id"][i])
        formatted_predictions.extend([{
            "image_id": img_id,
            "category_id": int(predictions["category_id"][i, pred_i]),
            "bbox": boxes[i, pred_i].cpu().tolist(),
            "score": float(predictions["score"][i, pred_i])
        } for pred_i in range(k_preds_per_image)])

    return formatted_predictions


@torch.no_grad()
def test(
    loader:DataLoader, 
    *, 
    model, 
    ds_info:dict,
    split:str,
    lvis_eval:bool,
    k_preds_per_image:int,
    cls_specific:bool=False):
    """
    Get mAP on a dataset
    """
    inf_start_time = time.time()
    # first need to get a query embedding for all possible labels/classes
    all_prompts = ds_info["best_7_imagenet_prompts"].copy()

    # need to ensure post-facto that none of the prompts will be truncated; max prompt length (in tokens) is 14
    for cls_name in all_prompts.keys():
        if cls_name == "monitor (computer equipment) computer monitor":
            all_prompts[cls_name] = [prompt.replace("monitor (computer equipment) computer monitor", "computer monitor") for prompt in all_prompts[cls_name]]
        elif cls_name == "peeler (tool for fruit and vegetables)":
            all_prompts[cls_name] = [prompt.replace("peeler (tool for fruit and vegetables)", "peeler for fruit and vegetables") for prompt in all_prompts[cls_name]]

    query_embeddings, n_prompts_for_each_class, ordered_class_names = get_test_queries(
        all_prompts, model.module.query_emb_dim, model.module.processor, model)

    classes_are_exclusive = ds_info["classes_are_exclusive"]

    assert all([isinstance(cat_id, int) for cat_id in ds_info["custom_cls_name_to_coco_cat_id"].values()])

    # now predict -> ensemble logits for each class -> get top k predictions -> format and append
    pred_i_to_cat_id = lambda i: ds_info["custom_cls_name_to_coco_cat_id"][ordered_class_names[i]]

    num_samples = loader.sampler.num_samples
    all_predictions = {
        "image_id": torch.zeros((num_samples,), dtype=torch.int) - 1.0,
        "category_id": torch.zeros((num_samples, k_preds_per_image), dtype=torch.int) - 1.0,
        "bbox": torch.zeros((num_samples, k_preds_per_image, 4), device=torch.device(dist.get_rank())) - 1.0,
        "score": torch.zeros((num_samples, k_preds_per_image)) - 1.0,
    } # [{image_id:int, category_id: int, bbox: [x, y, w, h], score: float}]

    preds_len = 0
    if dist.get_rank() == 0:
        print(torch.cuda.memory_summary(device=0))
    for batch_i, batch in enumerate(loader):
        images, image_ids = prepare_batch(batch, None, device=torch.device(dist.get_rank()), images_only=True)
        batch_len = images.size(0)

        images = images.to(torch.device(dist.get_rank()))
        image_features = model.module.get_image_features(images) # (B, num ViT tokens, ViT token dim)
        pred_logits = F.sigmoid(model.module.get_logits(image_features, query_embeddings)) # (B, num ViT tokens, sum(num prompts per class))

        pred_boxes = model.module.get_boxes(image_features) # (B, num ViT tokens, 4)

        # average the logits for each class's prompts
        pred_logits = torch.stack([
            torch.mean(chunk, dim=-1) for chunk in pred_logits.split(n_prompts_for_each_class, dim=-1)
        ], dim=-1)

        assert list(pred_logits.shape) == [images.size(0), pred_boxes.size(1), len(n_prompts_for_each_class)]

        # select top-k predictions; should be greater than the max number of objects per image
        top_scores, class_indices, top_boxes = get_top_k_preds(pred_logits, pred_boxes, k_preds_per_image, classes_are_exclusive) 

        # append predictions
        all_predictions["image_id"][preds_len:preds_len+batch_len] = image_ids
        all_predictions["bbox"][preds_len:preds_len+batch_len] = top_boxes
        all_predictions["category_id"][preds_len:preds_len+batch_len] = torch.tensor([[pred_i_to_cat_id(class_indices[i, pred_i]) for pred_i in range(class_indices.size(1))] for i in range(batch_len)], dtype=torch.int)
        all_predictions["score"][preds_len:preds_len+batch_len] = top_scores

        preds_len += batch_len

        if batch_i % 10 == 0:
            print_rank_0(f"{batch_i/len(loader) * 100:.2f}%")

    # chop off padding entries (DistributedSampler)
    for key in all_predictions:
        all_predictions[key] = all_predictions[key][:preds_len]

    # get all predictions in COCO format in a vectorized, batched fashion
    all_predictions = format_predictions(all_predictions, ds_info[split], k_preds_per_image)

    # run preds thru the evaluator
    pred_pth = f"tmp/preds/{socket.gethostname()}/pred_{dist.get_rank()}.json"
    os.makedirs(pred_pth.replace("/" + pred_pth.split("/")[-1], ""), exist_ok=True)
    with open(pred_pth, "w") as f:
        json.dump(all_predictions, f)

    dist.barrier()

    print_rank_0(f"Inference + formatting took {(time.time() - inf_start_time) / 60:.2f}min")

    results_pth = pred_pth.replace(pred_pth.split("/")[-1], "results.pkl")
    if dist.get_rank() == 0:
        all_pred_pth = merge_predictions()
        compute_mAP(all_pred_pth, results_pth, ds_info["json_paths"][split], lvis_eval, cls_specific=cls_specific)

    if cls_specific:
        exit(0)

    dist.barrier()

    with open(results_pth, "rb") as f:
        metrics = pickle.load(f)

    return metrics


def merge_predictions():
    all_pred_pth = f"tmp/preds/{socket.gethostname()}/all_pred.json"
    all_pred = []
    for i in range(dist.get_world_size()):
        with open(f"tmp/preds/{socket.gethostname()}/pred_{i}.json", "r") as f:
            all_pred.extend(json.load(f))

    with open(all_pred_pth, "w") as f:
        json.dump(all_pred, f)

    return all_pred_pth


def compute_mAP(pred_pth:str, results_pth:str, ann_pth:str, lvis_eval:bool, cls_specific:bool=False):
    print(pred_pth)
    start = time.time()
    if not lvis_eval:
        coco_gt = COCO(ann_pth)
        coco_dt = coco_gt.loadRes(pred_pth)

        if cls_specific:
            results = {}
            for catId in coco_gt.getCatIds():
                coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
                print("Evaluating for cat ID", catId)
                coco_eval.params.catIds = [catId]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                results[catId] = {k: v for k, v in zip(COCO_METRIC_NAMES, coco_eval.stats)}
            
            results_pth = f"results/cls_specific_accs/r{int(time.time())}.pkl"
        else:
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            results = {k: v for k, v in zip(COCO_METRIC_NAMES, coco_eval.stats)}
    else:
        lvis_gt = LVIS(ann_pth)
        lvis_dt = LVISResults(lvis_gt, pred_pth)
        lvis_eval = LVISEval(lvis_gt, lvis_dt, iou_type='bbox')
        lvis_eval.evaluate()
        lvis_eval.accumulate()
        lvis_eval.summarize()
        lvis_eval.print_results()
        results = lvis_eval.results

    print(f"mAP computation took: {time.time() - start:.2f}")

    with open(results_pth, "wb") as f:
        pickle.dump(results, f)

    print(f"Dumped results: {results_pth}")


def get_mAP(metrics:dict, lvis:bool=False):
    """
    Datasets can have different maxdets=n values, so we need to get the standard mAP metric reliably (without a KeyError)
    """
    if lvis:
        return {
            "all": metrics["AP"],
            "rare": metrics["APr"]
        }
    else:
        for key in metrics.keys():
            if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=" in key:
                return metrics[key]


def save_model(model, wandb_proj, wandb_run_name, ds_name, model_name, ds_id=None):
    if ds_id is None:
        model_path = f"models/state_dicts/{wandb_proj}/{wandb_run_name}_{ds_name}_{model_name}.pth"
        os.makedirs(f"models/state_dicts/{wandb_proj}", exist_ok=True)
    else:
        model_path = f"models/state_dicts/{wandb_proj}/cls_specific/{wandb_run_name}_{ds_name}_{model_name}/{ds_id}.pth"
        os.makedirs(f"models/state_dicts/{wandb_proj}/cls_specific/{wandb_run_name}_{ds_name}_{model_name}/", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print_rank_0(f"Saved model to '{model_path}'")


def log_class_specific_accs(cls_specific_accs, num_classes):
    data = [
        [
            i, 
            (cls_specific_accs[i]["correct"] / cls_specific_accs[i]["count"]),
            cls_specific_accs[i]["count"]
        ] for i in range(num_classes)
    ]

    table = wandb.Table(data=data, columns=["Class ID", "Accuracy", "Class Count"])
    wandb_log_rank_0({
        "cls_specific_accs": table
    })


"""
LR Schedulers
"""
class WarmupConstantScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            warmup_factor = self._step_count / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs

    def step(self):
        self._step_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineAnnealingScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=1e-10, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(WarmupCosineAnnealingScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            warmup_factor = self._step_count / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            cosine_decay_steps = self._step_count - self.warmup_steps
            annealing_steps = self.total_steps - self.warmup_steps
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_decay_steps / annealing_steps)) / 2
                    for base_lr in self.base_lrs]

    def step(self):
        self._step_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == "__main__":
    compute_mAP("../lvis_straggler.json", "none", "../data/LVIS/custom_val.json", True, False)