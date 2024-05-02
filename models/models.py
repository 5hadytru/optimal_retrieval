"""
Almost all code seen here is a torch version of code from the google scenic library,
especially the projects/owl_vit section of the repo
"""
from torch import nn
import torch.distributed as dist
import math
import torch
import torch.nn.functional as F 
import torchvision.transforms as T
import numpy as np
import time
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Union

# from importlib.machinery import SourceFileLoader
# gpu_matchers = SourceFileLoader("gpu_matcher", "models/gpu_matcher.py").load_module()

# custom types
ArrayDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, Tuple[torch.Tensor, torch.Tensor]]

HF_CACHE_DIR = "/work/itr176"

"""
Helper functions and classes
"""

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(
    boxes1,
    boxes2,
    all_pairs: bool = True,
    eps: float = 1e-6):
    """Computes IoU between two sets of boxes.

    Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
    right.

    Args:
    boxes1: Predicted bounding-boxes in shape [bs, n, 4].
    boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a different
        number of boxes if all_pairs is True.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
    eps: Epsilon for numerical stability.

    Returns:
    If all_pairs == True, returns the pairwise IoU cost matrix of shape
    [bs, n, m]. If all_pairs == False, returns the IoU between corresponding
    boxes. The shape of the return value is then [bs, n].
    """

    # First, compute box areas. These will be used later for computing the union.
    wh1 = boxes1[..., 2:] - boxes1[..., :2]
    area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

    wh2 = boxes2[..., 2:] - boxes2[..., :2]
    area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

    if all_pairs:
        # Compute pairwise top-left and bottom-right corners of the intersection
        # of the boxes.
        lt = torch.maximum(boxes1[..., :, None, :2],
                                    boxes2[..., None, :, :2])  # [bs, n, m, 2].
        rb = torch.minimum(boxes1[..., :, None, 2:],
                                    boxes2[..., None, :, 2:])  # [bs, n, m, 2].

        # intersection = area of the box defined by [lt, rb]
        wh = (rb - lt).clip(0.0)  # [bs, n, m, 2]
        intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

        # union = sum of areas - intersection
        union = area1[..., :, None] + area2[..., None, :] - intersection

        iou = intersection / (union + eps)
    else:
        # Compute top-left and bottom-right corners of the intersection between
        # corresponding boxes.
        assert boxes1.shape[1] == boxes2.shape[1], (
            'Different number of boxes when all_pairs is False')
        lt = torch.maximum(boxes1[..., :, :2],
                                    boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.minimum(boxes1[..., :, 2:], boxes2[..., :,
                                                            2:])  # [bs, n, 2]

        # intersection = area of the box defined by [lt, rb]
        wh = (rb - lt).clip(0.0)  # [bs, n, 2]
        intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]

        # union = sum of areas - intersection.
        union = area1 + area2 - intersection

        # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
        iou = intersection / (union + eps)

    return iou, union 


def generalized_box_iou(
        boxes1,
        boxes2,
        all_pairs: bool = True,
        eps: float = 1e-6):
    """Generalized IoU from https://giou.stanford.edu/.

    The boxes should be in [x, y, x', y'] format specifying top-left and
    bottom-right corners.

    Args:
        boxes1: Predicted bounding-boxes in shape [..., n, 4].
        boxes2: Target bounding-boxes in shape [..., m, 4].
        all_pairs: Whether to compute generalized IoU from between all-pairs of
        boxes or not. Note that if all_pairs == False, we must have m==n.
        eps: Epsilon for numerical stability.

    Returns:
        If all_pairs == True, returns a [bs, n, m] pairwise matrix, of generalized
        ious. If all_pairs == False, returns a [bs, n] matrix of generalized ious.
    """
    # Degenerate boxes gives inf / nan results, so do an early check.
    # TODO(b/166344282): Figure out how to enable asserts on inputs with jitting:
    # assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
    # assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()
    iou, union = box_iou(
        boxes1, boxes2, all_pairs=all_pairs, eps=eps)

    # Generalized IoU has an extra term which takes into account the area of
    # the box containing both of these boxes. The following code is very similar
    # to that for computing intersection but the min and max are flipped.
    if all_pairs:
        lt = torch.minimum(boxes1[..., :, None, :2],
                                boxes2[..., None, :, :2])  # [bs, n, m, 2]
        rb = torch.maximum(boxes1[..., :, None, 2:],
                                boxes2[..., None, :, 2:])  # [bs, n, m, 2]

    else:
        lt = torch.minimum(boxes1[..., :, :2],
                                boxes2[..., :, :2])  # [bs, n, 2]
        rb = torch.maximum(boxes1[..., :, 2:], boxes2[..., :,
                                                            2:])  # [bs, n, 2]

    # Now, compute the covering box's area.
    wh = (rb - lt).clip(0.0)  # Either [bs, n, 2] or [bs, n, m, 2].
    area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m].

    # Finally, compute generalized IoU from IoU, union, and area.
    # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
    return iou - (area - union) / (area + eps)


def weighted_box_l1_loss(
        pred: torch.Tensor,
        tgt: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        reduction: Optional[str] = None,
        tight: bool = True,
    ) -> torch.Tensor:
    """L1 loss for bounding box with optional reduction specified.

    Args:
        pred: Prediction boxes of shape (..., 4), where the last dimension has form
        (x_min, y_min, x_max, y_max).
        tgt: Target boxes of shape (..., 4), where the last dimension has form
        (x_min, y_min, x_max, y_max).
        weights: Weights to apply to the loss.
        reduction: Type of reduction, which is from [None, 'mean'].
        tight: If True, returns the vanilla L1 loss on the bounding box coordinates.
        If False, returns loose bounding-box L1 loss, where prediction edges only
        generate loss when they stretch outside the target box, but not when they
        are within it.

    Returns:
        reduction(jnp.abs(src - tgt)). 'mean' reduction takes the global mean. To
        use customized normalization use 'none' reduction and scale loss in the
        caller.
    """
    if pred.shape[-1] != 4:
        raise ValueError(
            f'The last dimension of the prediction boxes must be 4.'
            f' Got shape {pred.shape}.'
        )
    if tgt.shape[-1] != 4:
        raise ValueError(
            f'The last dimension of the target boxes must be 4.'
            f' Got shape {tgt.shape}.'
        )
    if tight:
        abs_diff = torch.abs(pred - tgt)
    else:
        xy1, xy2 = torch.split(pred - tgt, 2, dim=-1)
        xy1 = torch.minimum(xy1, 0.)
        xy2 = torch.maximum(xy2, 0.)
        abs_diff = torch.abs(torch.cat([xy1, xy2], dim=-1))
    if weights is not None:
        raise Exception("Currently not allowing weighted L1. See the scenic github repo.")
    if not reduction:
        return abs_diff
    elif reduction == 'mean':
        return abs_diff.mean()
    else:
        raise ValueError(f'Unknown reduction: {reduction}')


def sigmoid_cost(
        logit,
        *,
        focal_loss:bool = False,
        focal_alpha:float = None,
        focal_gamma:float = None
    ):
    """Computes the classification cost.

    Relevant code:
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/matcher.py#L76

    Args:
        logit: Sigmoid classification logit(s).
        focal_loss: Whether to apply focal loss for classification cost.
        focal_alpha: Alpha scaling factor for focal loss.
        focal_gamma: Gamma scaling factor for focal loss.

    Returns:
        Classification cost.
    """
    neg_cost_class = -F.logsigmoid(-logit)
    pos_cost_class = -F.logsigmoid(logit)
    if focal_loss:
        neg_cost_class *= (1 - focal_alpha) * F.sigmoid(logit)**focal_gamma
        pos_cost_class *= focal_alpha * F.sigmoid(-logit)**focal_gamma
    return pos_cost_class - neg_cost_class  # [B, N, C]


def hungarian_matcher(cost):
    """Computes Hungarian Matching given a single cost matrix.

    Relevant DETR code:
    https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

    Args:
        cost: Matching cost matrix of shape [N, M].

    Returns:
        Array of shape [min(N, M), 2] where each row contains a matched pair of
        indices into the rows (N) and columns (M) of the cost matrix.
    """
    # Matrix is transposed to maintain the convention of other matchers:
    col_ind, row_ind = linear_sum_assignment(cost.T)
    return np.stack([row_ind, col_ind]).astype(np.int32)


def matcher(cost:torch.Tensor, n_present_col:torch.Tensor, matching_fn=hungarian_matcher):
    """
    Adapted version of scenic/model_lib/matchers/common.slicer

    Maps matching_fn over examples after removing padding to speed up matching.

    Args:
    cost: Cost matrix or batch of cost matrices with any number of batch
        dimensions. Requires n_row >= n_col.
    n_present_col: Number of non-padding columns of the cost matrices, or None
        if padding should not be removed.
    matching_fn: A matching function that operates on a single cost matrix.

    Returns:
    Matchings of shape [batch, 2, n_col].

    Raises:
    ValueError if n_row < n_col and n_present_col is not None.
    """
    batch_shape = cost.shape[:-2]

    cost = cost.to(torch.float32).cpu().numpy()
    n_present_col = n_present_col.cpu().numpy()

    cost = cost.reshape(-1, *cost.shape[-2:])

    if n_present_col is None:
        matches = np.stack([matching_fn(c) for c in cost])
        return matches.reshape(*batch_shape, *matches.shape[1:])

    n_present_col = n_present_col.reshape(-1)
    assert cost.shape[0] == n_present_col.shape[0]

    batch, n_row, n_col = cost.shape
    if n_row < n_col:
        raise ValueError(
            f'Slicer requires that n_row ({n_row}) >= n_col ({n_col}).')

    eye = np.eye(n_row, dtype=np.bool_)
    matches = []
    for i in range(batch):
        present_col = int(max((n_present_col[i], 1)))  # One col even if all are padded.
        cost_m = cost[i, :, :present_col]  # Slicing should avoid a copy.

        row, col = matching_fn(cost_m)

        # Add padded matches (if padding was done correctly these can be random).
        unmatched_row = np.where(~eye[row].max(axis=0))[0]  # Faster than setdiff1d.
        unmatched_row = unmatched_row.astype(np.int32)
        unmatched_col = np.arange(present_col, n_col, dtype=np.int32)

        # Assume n_row >= n_col >= n_present_col.
        n_common = n_col - present_col
        unmatched_row = unmatched_row[:n_common]

        # Reconstruct the matching.
        row = np.concatenate([row, unmatched_row], axis=0)
        col = np.concatenate([col, unmatched_col], axis=0)

        matches.append(np.stack([row, col], axis=0))

    matches = np.stack(matches)
    matches = matches.reshape(*batch_shape, *matches.shape[1:])

    return torch.tensor(matches, dtype=torch.long)


@torch.vmap
def simple_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gathers `x` using the indices in `idx`.

    `output[i] = x[i, idx[i]]` . This simple gather operation assumes that the
    first dimension is the batch dimension. The indices index into the second
    dimension. The rest of the dimensions are copied as is from `x` into output.
    Note that the implementation below only handles a single element in the batch.
    `torch.vmap` extends this to the batch dimension.

    Args:
        x: Inputs of shape [bs, n, d].
        idx: An array of shape [bs, m] and dtype jnp.int32 or int64 that specifies
        indexes we want to gather from x.

    Returns:
        Gathered output of shape [bs, m, d].
    """
    return x[idx]


def focal_sigmoid_cross_entropy(
        logits: torch.Tensor,
        multi_hot_targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: Optional[float] = None,
        label_weights: Optional[torch.Tensor] = None,
        logits_normalized: bool = False,
        alpha: Optional[float] = 0.5,
        gamma: Optional[float] = 2.0
    ) -> torch.Tensor:
    """Computes focal softmax cross-entropy given logits and targets.

    Focal loss as defined in https://arxiv.org/abs/1708.02002. Assuming y is the
    target vector and p is the predicted probability for the class, then:

    p_t = p if y == 1 and 1-p otherwise
    alpha_t = alpha if y == 1 and 1-alpha otherwise

    Focal loss = -alpha_t * (1-p_t)**gamma * log(p_t)

    NOTE: this is weighted unnormalized computation of loss that returns the loss
    of examples in the batch. If you are using it as a loss function, you can
    use the normalilzed version as:
    ```
    unnormalized_loss = focal_sigmoid_cross_entropy(...)
    if weights is not None:
        normalization = weights.sum()
    else:
        normalization = np.prod(multi_hot_targets.shape[:-1])
    loss = jnp.sum(unnormalized_loss) / (normalization + 1e-8)
    ```

    Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    alpha: Balancing factor of the focal loss.
    gamma: Modulating factor of the focal loss.

    Returns:
    The loss of the examples in the given batch.
    """
    # Optionally apply label smoothing.
    if label_smoothing is not None:
        raise Exception("Must add label smoothing function")
        # multi_hot_targets = apply_label_smoothing(multi_hot_targets, label_smoothing)
    if logits_normalized:
        log_p, prob = logits, torch.exp(logits)
        log_not_p = torch.log((1 + 1e-6) - prob)
    else:
        log_p, log_not_p = F.logsigmoid(logits), F.logsigmoid(-logits)

    loss = -(multi_hot_targets * log_p + (1. - multi_hot_targets) * log_not_p)

    p_t = torch.exp(-loss)
    loss *= (1 - p_t)**gamma
    loss *= alpha * multi_hot_targets + (1 - alpha) * (1 - multi_hot_targets)

    if label_weights is not None:
        loss = loss * label_weights

    if weights is not None:
        raise Exception("Must add weight application function. See scenic repo")
        #loss = apply_weights(loss, weights)
    return loss


def weighted_unnormalized_sigmoid_cross_entropy(
    logits: torch.Tensor,
    multi_hot_targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    label_weights: Optional[torch.Tensor] = None,
    label_smoothing: Optional[float] = None,
    logits_normalized: bool = False) -> torch.Tensor:
    """Computes weighted sigmoid cross entropy given logits and targets.

    This also called Binary Cross-Entropy Loss and it measures the probability
    error in discrete classification tasks in which each class is independent and
    not mutually exclusive.
    This computes sum_(x,y) sigmoid-ce(x, y) for a single, potentially padded
    minibatch. If the minibatch is padded (that is it contains null examples)
    it is assumed that weights is a binary mask where 0 indicates that the
    example is null.

    Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
        This is the weight to apply to the loss computed for each example in the
        batch. Can be used to ignore padded examples in the batch.
    label_weights: None or array of shape broadcastable to the shape of logits.
        Typically this would be [num_classes] and is the weight to apply to each
        label.
    label_smoothing: Scalar to use to smooth the one-hot labels.
    logits_normalized: If True, the logits are assumed to be log probs.

    Returns:
    The sigmoid cross entropy of the examples in the given batch.
    """
    if logits.ndim != multi_hot_targets.ndim:
        raise ValueError(
            'Incorrect shapes. Got shape %s logits and %s multi_hot_targets' %
            (str(logits.shape), str(multi_hot_targets.shape)))

    # Optionally apply label smoothing.
    if label_smoothing is not None:
        raise Exception("Must implement label smoothing function. See scenic repo")
        multi_hot_targets = apply_label_smoothing(multi_hot_targets,
                                                    label_smoothing)

    if logits_normalized:
        log_p, prob = logits, torch.exp(logits)
        log_not_p = torch.log((1 + 1e-6) - prob)
    else:
        log_p, log_not_p = F.logsigmoid(logits), F.logsigmoid(-logits)

    loss = -(multi_hot_targets * log_p +
            (1. - multi_hot_targets) * log_not_p)

    if label_weights is not None:
        raise Exception("Only uncomment this line if you know what you're doing")
        # loss = loss * label_weights

    if weights is not None:
        raise Exception("Must implement weight application function. See scenic repo")
        #loss = apply_weights(loss, weights)

    return loss


class OWLViTLoss(nn.Module):
    """ 
    Taken from the detr repo then adapted:
        - Binary (multiclass) CE for classification
        - Optional focal CE
        - Optional per-sample normalization (by # objects) as opposed to per-batch
        - Target unpadding and organizing

    The loss is computed in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        *,
        use_focal_loss: bool=True,
        cost_class: float = 1, 
        cost_bbox: float = 1, 
        cost_giou: float = 1,
        alpha:float=0.3, 
        gamma:int=2,
        losses_and_metrics=["labels", "boxes"],
        per_image_norm=False
    ):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cost_class_coeff = cost_class
        self.cost_bbox_coeff = cost_bbox 
        self.cost_giou_coeff = cost_giou

        self.loss_terms_weights = {
            "loss_class": 1.0,
            "loss_bbox": 1.0,
            "loss_giou": 1.0
        }

        self.losses_and_metrics = losses_and_metrics
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.per_image_norm = per_image_norm


    def labels_losses_and_metrics(
            self,
            outputs: ArrayDict,
            batch: ArrayDict,
            indices: torch.Tensor,
            log: bool = False
        ) -> Tuple[ArrayDict, MetricsDict]:
        """Classification loss.

        Args:
        outputs: Model predictions. For the purpose of this loss, outputs must
        have key 'pred_logits'. outputs['pred_logits'] is a nd-array of the
        predicted logits of shape [batch-size, num-objects, num-classes].
        batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, label dict must
        have key 'labels', which the value is an int nd-array of labels with
        shape [batch_size, num_boxes, num_classes + 1]. Since the number of
        boxes (objects) in each example in the batch could be different, the
        input pipeline might add padding boxes to some examples. These padding
        boxes are identified based on their class labels. So if the class label
        is `0`, i.e., a one-hot vector of [1, 0, 0, ..., 0], the box/object is a
        padding object and the loss computation will take that into account. The
        input pipeline also pads the partial batches (last batch of eval/test
        set with num_example < batch_size). batch['batch_mask'] is used to
        identify padding examples which is incorporated to set the weight of
        these examples to zero in the loss computations.
        indices: Matcher output of shape [batch-size, 2, num-objects] which
        conveys source to target pairing of objects.
        log: If true, return classification accuracy as well.

        Returns:
        loss: Dict with 'loss_class' and other model specific losses.
        metrics: Dict with 'loss_class' and other model specific metrics.
        """
        assert 'pred_logits' in outputs
        assert 'label' in batch

        batch_weights = None
        losses = {
            "each": {},
            "mean": {}
        }
        metrics = {}
        targets = batch['label']["labels"]
        denom = batch["label"]["cls_denom"]

        src_logits = outputs['pred_logits']

        # Apply the permutation communicated by indices.
        src_logits = simple_gather(src_logits, indices[:, 0]) # (B,N,C)
        tgt_labels = simple_gather(targets, indices[:, 1]) # (B,N,C)

        unnormalized_loss_class, pos_labels_per_img = self._compute_per_example_class_loss( # (B,N), (B). denom is the number of positive labels for each image
            tgt_labels=tgt_labels,
            src_logits=src_logits,
            batch_weights=batch_weights,
        )

        metrics['loss_class'] = (unnormalized_loss_class.sum(), pos_labels_per_img.sum())

        if not self.per_image_norm:
            with torch.no_grad():
                per_image_loss_class = unnormalized_loss_class.sum(dim=1)
                per_image_normalized_loss_class = (per_image_loss_class / torch.maximum(pos_labels_per_img, torch.tensor(1.)))
            mean_normalized_loss_class = unnormalized_loss_class.sum() / denom
        else:
            per_image_loss_class = unnormalized_loss_class.sum(dim=1)
            per_image_normalized_loss_class = (per_image_loss_class / torch.maximum(pos_labels_per_img, torch.tensor(1.)))
            mean_normalized_loss_class = per_image_normalized_loss_class.mean()
            
        losses["each"]['loss_class'] = per_image_normalized_loss_class.detach()
        losses["mean"]['loss_class'] = mean_normalized_loss_class

        # Not using this code, so not changing it to torch
        # if log:
        #     # Class accuracy for non-padded (label != 0) labels
        #     not_padded = tgt_labels[:, :, 0] == 0
        # if batch_weights is not None:
        #     not_padded = not_padded * jnp.expand_dims(batch_weights, axis=1)
        #     num_correct_no_pad = model_utils.weighted_correctly_classified(
        #         src_logits[..., 1:], tgt_labels[..., 1:], weights=not_padded
        #     )
        # metrics['class_accuracy_not_pad'] = (num_correct_no_pad, not_padded.sum())

        # Sum metrics and normalizers over all replicas.
        # for k, v in metrics.items():
        #     metrics[k] = psum_metric_normalizer(v)

        return losses, metrics


    def _compute_per_example_class_loss(
            self,
            *,
            tgt_labels: torch.Tensor,
            src_logits: torch.Tensor,
            batch_weights: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the unnormalized per-example classification loss and denom."""
        loss_kwargs = {
            'weights': batch_weights,
        }
        if self.use_focal_loss:
            loss_kwargs['gamma'] = self.gamma
            loss_kwargs['alpha'] = self.alpha
            loss_fn = focal_sigmoid_cross_entropy
        else:
            loss_fn = weighted_unnormalized_sigmoid_cross_entropy

        # Don't compute loss for the padding index.
        unnormalized_loss_class = loss_fn(
            src_logits[..., 1:], 
            tgt_labels[..., 1:], 
            **loss_kwargs
        )
        # Sum losses over all classes. The unnormalized_loss_class is of shape
        # [bs, 1 + max_num_boxes, num_classes], and after the next line, it becomes
        # [bs, 1 + max_num_boxes].
        unnormalized_loss_class = torch.sum(unnormalized_loss_class, dim=-1)

        # Normalize by number of "true" labels after removing padding label.
        denom = tgt_labels[..., 1:].sum(dim=(1, 2))

        if batch_weights is not None:
            denom *= batch_weights

        return unnormalized_loss_class, denom


    def boxes_losses_and_metrics(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      indices: torch.Tensor) -> Tuple[ArrayDict, MetricsDict]:
        """Bounding box losses: L1 regression loss and GIoU loss.

        Args:
        outputs: dict; Model predictions. For the purpose of this loss, outputs
            must have key 'pred_boxes'. outputs['pred_boxes'] is a nd-array of the
            predicted box coordinates in (cx, cy, w, h) format. This nd-array has
            shape [batch-size, num-boxes, 4].
        batch: dict; that has 'inputs', 'batch_mask' and, 'label' (ground truth).
            batch['label'] is a dict. For the purpose of this loss, batch['label']
            must have key 'boxes', which the value has the same format as
            outputs['pred_boxes']. Additionally in batch['label'], key 'labels' is
            required that should match the specs defined in the member function
            `labels_losses_and_metrics`. This is to decide which boxes are invalid
            and need to be ignored. Invalid boxes have class label 0.
        indices: list[tuple[nd-array, nd-array]]; Matcher output which conveys
            source to target pairing of objects.

        Returns:
        loss: dict with keys 'loss_bbox', 'loss_giou'. These are
            losses averaged over the batch. Therefore they have shape [].
        metrics: dict with keys 'loss_bbox' and 'loss_giou`.
            These are metrics psumed over the batch. Therefore they have shape [].
        """
        targets = batch['label']
        losses = {
            "each": {},
            "mean": {}
        }
        metrics = {}
        batch_weights = None

        src_boxes = simple_gather(outputs['pred_boxes'], indices[:, 0])
        tgt_boxes = simple_gather(targets['boxes'], indices[:, 1])
        tgt_labels = targets['labels']
        denom = targets["box_denom"]

        # Some of the boxes are padding. We want to discount them from the loss.
        n_labels_per_instance = torch.sum(tgt_labels[..., 1:], dim=-1)
        tgt_not_padding = n_labels_per_instance > 0  # [B, M]

        # tgt_is_padding has shape [batch-size, num-boxes].
        # Align this with the model predictions using simple_gather.
        tgt_not_padding = simple_gather(tgt_not_padding, indices[:, 1])

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        unnormalized_loss_giou = 1 - generalized_box_iou(
            src_boxes_xyxy, tgt_boxes_xyxy, all_pairs=False)

        unnormalized_loss_bbox = weighted_box_l1_loss(
            src_boxes_xyxy,
            tgt_boxes_xyxy,
            weights=batch_weights,
        ).sum(dim=2)

        boxes_per_img = tgt_not_padding.sum(dim=1) # (B)
        if batch_weights is not None:
            raise Exception("Must add weight application function. See scenic repo")
            # denom *= batch_weights
            # unnormalized_loss_giou = model_utils.apply_weights(
            #     unnormalized_loss_giou, batch_weights)

        unnormalized_loss_bbox *= tgt_not_padding
        unnormalized_loss_giou *= tgt_not_padding

        if not self.per_image_norm:
            # Normalize by number of boxes in batch.
            with torch.no_grad():
                each_normalized_loss_bbox = (unnormalized_loss_bbox.sum(dim=1) / torch.maximum(boxes_per_img, torch.tensor(1.)))
                each_normalized_loss_giou = (unnormalized_loss_giou.sum(dim=1) / torch.maximum(boxes_per_img, torch.tensor(1.)))
            mean_normalized_loss_bbox = unnormalized_loss_bbox.sum() / denom
            mean_normalized_loss_giou = unnormalized_loss_giou.sum() / denom
        else:  # Normalize by number of boxes in image.
            denom = torch.maximum(boxes_per_img, torch.tensor(1.))

            each_normalized_loss_bbox = (unnormalized_loss_bbox.sum(dim=1) / denom)
            each_normalized_loss_giou = (unnormalized_loss_giou.sum(dim=1) / denom)

            mean_normalized_loss_bbox = each_normalized_loss_bbox.mean()
            mean_normalized_loss_giou = each_normalized_loss_giou.mean()

        losses['each']['loss_bbox'] = each_normalized_loss_bbox.detach()
        losses['each']['loss_giou'] = each_normalized_loss_giou.detach()
        
        losses['mean']['loss_bbox'] = mean_normalized_loss_bbox
        losses['mean']['loss_giou'] = mean_normalized_loss_giou

        metrics['loss_bbox'] = (mean_normalized_loss_bbox.detach(), torch.tensor(1.).to(mean_normalized_loss_bbox.device))
        metrics['loss_giou'] = (mean_normalized_loss_giou.detach(), torch.tensor(1.).to(mean_normalized_loss_giou.device))

        # Sum metrics and normalizers over all replicas.
        # for k, v in metrics.items():
        #     metrics[k] = psum_metric_normalizer(v) 

        return losses, metrics


    @torch.no_grad()
    def compute_cost_matrix(self, outputs:dict, targets:dict, use_focal_loss:bool, focal_alpha, focal_gamma):
        """ Computes 3D cost matrix -> performs the matching

        Params:
            outputs (dict):
                - pred_boxes: (batch size, num ViT tokens, 4)
                - pred_logits: (batch size, num ViT tokens, max prompts)
            targets (dict):
                - boxes: (batch size, max instances, 4)
                - labels: (batch size, max instances, max prompts + 1) # the +1 is for the padding dimension

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert outputs["pred_logits"].size(0) == outputs["pred_boxes"].size(0) == targets["labels"].size(0) == targets["boxes"].size(0), f'{outputs["pred_logits"].size()} == {outputs["pred_boxes"].size()} == {targets["labels"].size()} == {targets["boxes"].size()}'

        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]
        tgt_bbox = targets["boxes"]
        tgt_labels = targets["labels"].to(outputs["pred_logits"].dtype)

        # Number of non-padding labels for each of the target instances.
        n_labels_per_instance = tgt_labels[...,1:].sum(dim=-1)
        mask = n_labels_per_instance > 0  # [B, M]

        # Make sure padding target is 0 for instances with other labels.
        tgt_labels = torch.cat([(~mask).unsqueeze(-1), tgt_labels[...,1:]], dim=-1)

        cost_class = sigmoid_cost(  # (B,N,C)
            out_logits,
            focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

        # Resulting shape is [B, N, M].
        # Note that we do *not* normalize by the number of per-target instances
        cost_class = torch.einsum('bnl,bml->bnm', cost_class, tgt_labels)

        cost = self.cost_class_coeff * cost_class

        # Compute the L1 cost between boxes
        diff = torch.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])  # [B, N, M, 4]
        cost_bbox = torch.sum(diff, axis=-1)  # [B, N, M]
        cost += self.cost_bbox_coeff * cost_bbox
        
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        cost += self.cost_giou_coeff * cost_giou

        mask = mask[:, None] # (B,1,M)

        # Determine mask value dynamically.
        cost_mask_value = torch.amax(torch.where(mask, cost, -1e10), dim=(1, 2)) # (B); max cost for each image

        # Special case.
        all_masked = torch.all(torch.all(~mask, dim=1), dim=1) # (B); whether all instances are masked
        cost_mask_value = torch.where(~all_masked, cost_mask_value, 1.0)
        cost_mask_value = cost_mask_value[:, None, None] * 1.1 + 10.0 # (B,1,1)

        cost = cost * mask + (1.0 - mask.to(cost.dtype)) * cost_mask_value # replace masked cost values with cost_mask_value

        # Guard against NaNs and Infs.
        cost_mask_value = cost_mask_value.expand_as(cost)
        cost_is_nan = torch.isnan(cost)
        cost_is_inf = torch.isinf(cost)
        cost[cost_is_nan] = cost_mask_value[cost_is_nan]
        cost[cost_is_inf] = cost_mask_value[cost_is_inf]

        # Compute the number of unpadded columns for each batch element. It is assumed
        # that all padding is trailing padding.
        max_num_boxes = tgt_labels.shape[1] # M

        n_cols = torch.where(
            torch.any(mask, dim=1).unsqueeze(1), # (B,1,M)
            torch.arange(1, max_num_boxes + 1).unsqueeze(0).to(mask.device), # (1,M)
            torch.zeros(1, max_num_boxes).to(mask.device)
        ) 

        n_cols = torch.max(n_cols, dim=2).values

        return cost, n_cols
    

    @property
    def loss_and_metrics_map(self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
        """Returns a dict that lists all losses for this model."""
        return {
            'labels': self.labels_losses_and_metrics,
            'boxes': self.boxes_losses_and_metrics,
        }


    def get_losses_and_metrics(
            self, 
            loss: str, 
            outputs: ArrayDict,
            batch: ArrayDict, 
            indices: torch.Tensor,
            **kwargs: Any
        ):
        assert loss in self.loss_and_metrics_map, f'Unknown loss {loss}.'
        return self.loss_and_metrics_map[loss](outputs, batch, indices, **kwargs)


    def get_metrics(self, metrics_dict: MetricsDict) -> MetricsDict:
        """Arrange loss dictionary into a metrics dictionary."""
        metrics = {}
        # Some metrics don't get scaled, so no need to keep their unscaled version,
        # i.e. those that are not in self.loss_terms_weights.keys()
        for k, v in metrics_dict.items():
            loss_term = self.loss_terms_weights.get(k)
            if loss_term is not None:
                metrics[f'{k}_unscaled'] = v
                metrics[k] = (loss_term * v[0], v[1])
            else:
                metrics[k] = v

        return metrics


    def forward(self, batch:dict):
        """ This performs the loss computation.
        Parameters:
            batch: dict of tensors containing 'pred_logits', 'pred_boxes', 'boxes', and 'labels' (one/multi-hot for each instance and prompt)
        """
        batch = batch.copy()

        # Append an instance with "padding" label (i.e., "0" as the class label).
        # Shape is [batch, num_instances, num_classes]. This is necessary because
        # the matching code requires at least one padding instance, to which
        # unmatched instances will be assigned.
        label_shape = batch['labels'].shape
        num_classes = label_shape[-1]
        instance = F.one_hot(torch.tensor(0), num_classes)
        reshape_shape = (1,) * (len(label_shape) - 1) + (num_classes,)
        broadcast_shape = label_shape[:-2] + (1, num_classes)
        instance = torch.broadcast_to(
            torch.reshape(instance, reshape_shape), broadcast_shape).to(batch['labels'].device)
        batch['labels'] = torch.cat(
            [batch['labels'], instance], 
            dim=-2
        )

        instance = torch.zeros_like(batch['boxes'][..., :1, :])
        batch['boxes'] = torch.cat(
            [batch['boxes'], instance], dim=-2)

        cost, n_cols = self.compute_cost_matrix(
            {
                "pred_logits": batch["pred_logits"],
                "pred_boxes": batch["pred_boxes"]
            },
            {
                "boxes": batch["boxes"],
                "labels": batch["labels"]
            },
            self.use_focal_loss, self.alpha, self.gamma
        )

        # match predictions to target instances
        matches = matcher(cost, n_cols)

        if not isinstance(matches, (list, tuple)):
            # Ensure matches come as a sequence.
            matches = [matches]

        # Pad matches if the matching is not complete (i.e. the number of
        # predicted instances is larger than the number of gt instances).
        num_pred = batch['pred_logits'].shape[-2]

        def pad_matches(match):
            batch_size, _, num_matched = match.shape  # [B, 2, M]
            if num_pred > num_matched:

                def get_unmatched_indices(row, ind):
                    row[ind] = 1
                    return torch.topk(torch.logical_not(row).to(torch.int8), k=num_pred - num_matched)

                get_unmatched_indices = torch.vmap(get_unmatched_indices)

                indices = torch.zeros((batch_size, num_pred), dtype=torch.bool)
                _, indices = get_unmatched_indices(indices, match[:, 0, :])
                indices = indices.unsqueeze(1)

                padding = torch.cat(
                    [indices, torch.full(indices.shape, fill_value=num_matched - 1)],
                    dim=1)
                return torch.cat([match, padding], dim=-1)
            return match

        matches = [pad_matches(match) for match in matches]

        indices = matches[0]

        # Compute all the requested losses and metrics.
        loss_dict = {"each": {}, "mean": {}}
        metrics_dict = {}
        outputs = {
            "pred_boxes": batch["pred_boxes"], 
            "pred_logits": batch["pred_logits"]
        }
        batch = {
            "label": {
                "boxes": batch["boxes"],
                "labels": batch["labels"],
                "cls_denom": batch["cls_denom"],
                "box_denom": batch["box_denom"]
            }
        }
        for loss_name in self.losses_and_metrics:
            loss, metrics = self.get_losses_and_metrics(loss_name, outputs, batch, indices)
            loss_dict["each"].update(loss["each"])
            loss_dict["mean"].update(loss["mean"])
            metrics_dict.update(metrics)

        # Compute the total loss by combining loss_dict with loss_terms_weights.
        mean_losses = []
        for k, v in loss_dict["mean"].items():
            if k in self.loss_terms_weights:
                mean_losses.append(self.loss_terms_weights[k] * v)
        total_mean_loss = sum(mean_losses)

        loss_eaches = []
        for k, v in loss_dict["each"].items():
            if k in self.loss_terms_weights:
                loss_eaches.append(self.loss_terms_weights[k] * v)
        total_loss_each = sum(loss_eaches)

        # Process metrics dictionary to generate final unnormalized metrics.
        metrics = self.get_metrics(metrics_dict)
        metrics['total_loss'] = (mean_losses, 1)

        return total_mean_loss, total_loss_each.detach(), metrics

"""
Models
"""

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_feature_dim(self):
        return self.num_features


class OWL_ViT(BaseModel):
    """
    Take the OWL-ViT model and continue DETR-style learning (https://arxiv.org/abs/2205.06230)
    NOTE: input_ids and attention_mask are padded by zeroes while labels and boxes are padded by -1 
    """
    def __init__(self, hf_model, processor, num_features, query_emb_dim, feature_map_dim):
        super().__init__()
        
        self.num_features = num_features
        self.hf_model = hf_model
        self.processor = processor
        self.query_emb_dim = query_emb_dim
        self.loss_func = OWLViTLoss()
        self.feature_map_dim = feature_map_dim


    def precompute_box_bias(self, batch_size):
        """
        HF model had a glaring inefficiency; computed box bias for each call to box_predictor
        """
        self.hf_model.box_bias = self.hf_model.compute_box_bias(torch.zeros((batch_size, self.feature_map_dim, self.feature_map_dim, self.num_features))).to(self.hf_model.device)


    def forward(self, images, input_ids, attention_mask, **kwargs):
        """
        Using this for training; during val it will just be image_forward
        Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py to handle different number of queries for each image
        """
        forward_dict = self.hf_model(input_ids=input_ids.view(-1, input_ids.size(-1)), pixel_values=images, attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        return {
            "pred_boxes": forward_dict["pred_boxes"],
            "pred_logits": forward_dict["logits"]
        }


    def get_image_features(self, images, return_cls_token:bool=False):
        """
        Get image features such that they are ready to be passed thru the OwlViTClassPredictionHead with
        text embeddings

        NOTE: these are the image features right before being passed thru the class and box prediction projectors
        """
        vision_outputs = self.hf_model.owlvit.vision_model(
            pixel_values=images,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        last_hidden_state = vision_outputs[0]

        image_embeds = self.hf_model.owlvit.vision_model.post_layernorm(last_hidden_state)

        cls_token = image_embeds[:, :1, :].squeeze(1).detach().clone()

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.hf_model.layer_norm(image_embeds)

        if return_cls_token:
            return image_embeds, cls_token
        else:
            return image_embeds


    def get_text_embeddings(self, input_ids, attention_mask):
        """
        Get text embeddings such that they are ready to be passed on to the class prediction head
        """
        text_outputs = self.hf_model.owlvit.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        text_embeds = text_outputs[1] # pooler output
        text_embeds = self.hf_model.owlvit.text_projection(text_embeds)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)

        return text_embeds


    def get_boxes(self, image_features):
        """
        Pass the image features (pre class projection) thru the box prediction head
        """
        pred_boxes = self.hf_model.box_predictor(image_features)

        return pred_boxes


    def get_logits(self, image_features, text_embeddings, return_embeds=False, query_mask=None):
        """
        Pass the image features (pre class projection) and text embeddings (post class projection) thru the class
        prediction head

        NOTE: must provide query mask if you have padding. See HF repo (https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py#L1695)).
        Unnecessary here since I am only using this method for testing.
        """
        (pred_logits, class_embeds) = self.hf_model.class_predictor(image_features, text_embeddings, query_mask)

        return pred_logits if not return_embeds else (pred_logits, class_embeds)

"""
Specific models
"""
def OWL_B32():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    num_features = 768
    query_emb_dim = 512
    feature_map_dim = 24

    return OWL_ViT(model, processor, num_features, query_emb_dim, feature_map_dim)

def OWL_B16():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
    num_features = 768
    query_emb_dim = 512
    feature_map_dim = 48

    return OWL_ViT(model, processor, num_features, query_emb_dim, feature_map_dim)

def OWL_L14():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
    num_features = 1024
    query_emb_dim = 768
    feature_map_dim = 60

    return OWL_ViT(model, processor, num_features, query_emb_dim, feature_map_dim)


def get_cls_denom(labels:torch.Tensor, float_dtype):
    if labels.numel() == 0:
        cls_denom = torch.tensor([0.]).to(torch.device(0))
        contributed = torch.tensor(0.).to(torch.device(0))
    else:
        cls_denom = torch.tensor([labels[...,1:].sum()]).to(float_dtype).to(torch.device(0))
        contributed = torch.tensor(1.).to(torch.device(0))

    cls_denom /= contributed
    cls_denom = torch.maximum(cls_denom, torch.tensor([1.]).to(torch.device(0)))

    return cls_denom
    

def get_box_denom(boxes:torch.Tensor, labels:torch.Tensor):
    if boxes.numel() == 0:
        box_denom = torch.tensor([0.]).to(torch.device(0))
        contributed = torch.tensor(0.).to(torch.device(0))
    else:
        n_labels_per_instance = torch.sum(labels[..., 1:], dim=-1)
        box_denom = torch.tensor([torch.sum(n_labels_per_instance > 0, dtype=boxes.dtype)]).to(torch.device(0))
        contributed = torch.tensor(1.).to(torch.device(0))

    box_denom /= contributed
    box_denom = torch.maximum(box_denom, torch.tensor([1.]).to(torch.device(0)))

    return box_denom


def add_denoms(y:dict):
    """
    Given local boxes and labels, add shared (across devices) denominators for box and label loss normalization
    The shared value of cls_denom is the average number of positive labels (1s) per-image across devices
    The shared value of box_denom is the average number of boxes per-image across devices
    """
    cls_denom = get_cls_denom(y["labels"], y["boxes"].dtype)
    box_denom = get_box_denom(y["boxes"], y["labels"])

    y["cls_denom"] = cls_denom
    y["box_denom"] = box_denom


if __name__ == "__main__":
    device = torch.device("cuda:0")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    batch_size = 1
    img_size = 840
    dtype = torch.bfloat16

    inputs = {
        "images": torch.randn((batch_size, 3, img_size, img_size), dtype=dtype).to(device),
        "input_ids": torch.ones((batch_size, 126, 16), dtype=torch.long).to(device),
        "attention_mask": torch.zeros((batch_size, 126, 16), dtype=torch.long).to(device)
    }

    torch.cuda.reset_max_memory_allocated(device=device)

    y = {
        'boxes': torch.randn((batch_size,2023,4), dtype=dtype).to(device),
        'labels': torch.ones((batch_size,2023,126), dtype=torch.int).to(device)
    }

    # inputs["input_ids"][0][4] = 0 * torch.ones((16,), dtype=torch.long)
    # inputs["input_ids"][1][6] = 0 * torch.ones((16,), dtype=torch.long)

    model = OWL_L14().to(device).to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    model.precompute_box_bias(batch_size)

    outputs = model(**inputs)
    l1 = outputs["pred_logits"]

    model = model.to(torch.bfloat16)
    torch.cuda.empty_cache()

    query_embeds = model.get_text_embeddings(
        inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze())
    image_features, cls_token = model.get_image_features(inputs["images"], return_cls_token=True)
    pred_boxes = model.get_boxes(image_features)
    l2, embeds = model.get_logits(image_features, query_embeds, return_embeds=True)

    print(image_features.size(), embeds.size())

    exit(0)

    """
    query_embeds = model.get_text_embeddings(
        inputs["input_ids"].view(-1, inputs["input_ids"].size(-1)), 
        inputs["attention_mask"].view(-1, inputs["input_ids"].size(-1))
    )
    query_embeds = query_embeds.view(inputs["input_ids"].size(0), inputs["input_ids"].size(1), query_embeds.size(-1))
    image_features = model.get_image_features(inputs["images"], return_cls_token=False) # (B, num ViT tokens, ViT token dim)
    m_pred_boxes = model.get_boxes(image_features)
    m_pred_logits = model.get_logits(image_features, query_embeds) # (B, num ViT tokens, sum(num prompts per class))
    _, m_loss_each, __ = model.loss_func({"pred_logits": m_pred_logits, "pred_boxes": m_pred_boxes} | y)
    """
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        add_denoms(y)
        outputs = model(**inputs)
        l_pred_logits = outputs["pred_logits"]
        l_pred_boxes = outputs["pred_boxes"]
        loss_mean, _, __ = model.loss_func({"pred_logits": l_pred_logits, "pred_boxes": l_pred_boxes} | y)
    loss_mean.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    max_memory_used = torch.cuda.max_memory_allocated(device=device)
    print(f"Maximum GPU memory allocated during script execution: {max_memory_used / 1024**2:.2f} MB")

    exit(0)

    loss = model.loss_func(outputs | y)

    # from torch.func import functional_call, vmap, grad

    # params = {k: v.detach() for k, v in model.named_parameters()}
    # buffers = {k: v.detach() for k, v in model.named_buffers()}

    # def compute_loss(params, buffers, images, input_ids, attention_mask, labels, boxes):
    #     images = images.unsqueeze(0)
    #     input_ids = input_ids.unsqueeze(0)
    #     attention_mask = attention_mask.unsqueeze(0)
    #     labels = labels.unsqueeze(0)
    #     boxes = boxes.unsqueeze(0)

    #     predictions = functional_call(model, (params, buffers), (images, input_ids, attention_mask))
    #     loss_mean, loss_each, metrics, nan_cost, inf_cost = model.loss_func(predictions | {"boxes": boxes, "labels": labels})
    #     print("Made it")
    #     exit(0)
    #     return loss_mean

    # ft_compute_grad = grad(compute_loss)

    # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0, 0, 0, 0))

    # ft_per_sample_grads = ft_compute_sample_grad(
    #     params, 
    #     buffers, 
    #     inputs["images"], 
    #     inputs["input_ids"],
    #     inputs["attention_mask"],
    #     y["labels"],
    #     y["boxes"]
    # )
    

    memory_before_forward = torch.cuda.memory_allocated()
    outputs = model(**inputs)
    memory_after_forward = torch.cuda.memory_allocated()

    print(f"Memory consumed during forward pass: {(memory_after_forward - memory_before_forward) / 1024 ** 3:.1f} GB; before: {memory_before_forward  / 1024 ** 3:.1f} after: {memory_after_forward  / 1024 ** 3:.1f}")

    memory_before_loss = torch.cuda.memory_allocated()
    memory_after_loss = torch.cuda.memory_allocated()
    print(f"Memory consumed during loss: {(memory_after_loss - memory_before_loss) / 1024 ** 2} MB")

    memory_before_backward = torch.cuda.memory_allocated()
    memory_after_backward = torch.cuda.memory_allocated()
    print(f"Memory consumed during backward pass: {(memory_after_backward - memory_before_backward) / 1024 ** 3:.1f} GB; after: {memory_after_backward  / 1024 ** 3:.1f}")

    memory_before_step = torch.cuda.memory_allocated()
    optimizer.step()
    memory_after_step = torch.cuda.memory_allocated()
    print(f"Memory consumed during optimizer step: {(memory_after_step - memory_before_step) / 1024 ** 3:.1f} GB; after: {memory_after_step  / 1024 ** 3:.1f}")

    print(torch.cuda.memory_summary())