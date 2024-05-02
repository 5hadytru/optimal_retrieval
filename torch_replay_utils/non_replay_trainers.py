from abc import ABC, abstractmethod
from torch_replay_utils import opt_utils
import torch


class NonReplayTrainer(ABC):
    def __init__(self, device, cfg):
        self.cfg = cfg
    
    @abstractmethod
    def train_batch(self, *, inputs:dict, y:dict, model, optimizer, epoch):
        pass

    def post_epoch_logs(self):
        """
        Logs for after every epoch
        """
        pass

    @abstractmethod
    def init_post_epoch_data(self):
        """
        Initialize the dictionary containing post-epoch wandb logs
        """
        pass


class Base_SGD(NonReplayTrainer):
    """
    Object which replaces replay_utils Trainers when doing standard offline training
    """
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        self.forward_pass = opt_utils.Basic(cfg)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def train_batch(self, *, inputs:dict, y:dict, model, optimizer, epoch):
        loss_mean = self.forward_pass(inputs, y, model)
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    def init_post_epoch_data(self):
        """
        Initialize the dictionary containing post-epoch wandb logs
        """
        pass