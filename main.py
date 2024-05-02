import torch
import numpy as np
import wandb
import time
import gc
import os, gc
import tqdm
import random
import copy, pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import omegaconf
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist
import os, sys, socket

# import custom modules
from util_funcs import data_utils, model_utils, main_utils
from importlib.machinery import SourceFileLoader
models = SourceFileLoader("models", "models/models.py").load_module()
from torch_replay_utils.replay_trainers import C_ASER, Random, SWIL
from torch_replay_utils.non_replay_trainers import Base_SGD


proxy_dataset_configs = {
    "all": ["LVIS", "O365"],
    "custom": ["O365"],
    "GLIP": ["O365", "LVIS"]
}

no_FT_datasets = ["VehiclesOpenImages", "CottontailRabbits", "NorthAmericaMushrooms"]
no_test_datasets = ["VehiclesOpenImages", "CottontailRabbits", "NorthAmericaMushrooms"]

SWIL_DATASET_MAP = {
    0: ['pothole', "ChessPieces", "WildfireSmoke"],
    1: ['pothole', "ChessPieces", "WildfireSmoke", "OxfordPets", "EgoHands"],
    2: ['pothole', "ChessPieces", "WildfireSmoke", "OxfordPets", "EgoHands", "AmericanSignLanguageLetters", "dice"]
}

max_batch_sizes = {
    "OWL_B16": {
        "eval": 36,
        "train": 8
    },
    "OWL_L14": {
        "eval": 24,
        "train": 4
    },
    "v100": {
        "eval": 16,
        "train": 2
    }
}

get_batch_size = lambda model_id, split, cfg: max_batch_sizes[model_id][split] if not cfg.main.using_v100 else max_batch_sizes["v100"][split]

# parse cmd line
# parser = argparse.ArgumentParser(description='Specify master params')

# parser.add_argument('--config_path', type=str, required=True)

# master_args = vars(parser.parse_args())


# global variables
RANK, WORLD_SIZE = None, None

k_preds_per_image = lambda ds_name: 300 if ds_name not in ["DroneControl", "PascalVOC", "EgoHands", "brackishUnderwater"] else 100

TRAINERS = {
    "Random": Random,
    "RandomSUB": Random,
    "RandomUB": Random,
    "C_ASER": C_ASER,
    "LWUB": Random,
    "GRASP": Random,
    "SWIL": SWIL
}


def launch_main(rank:int):
    global RANK
    RANK = rank
    main()


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def init_trainer(device, cfg):
    """
    Get a Trainer object, which allows main.py to just call trainer.train_batch
    """
    if cfg.main.retrieval_strat is None:
        return Base_SGD(device, cfg)
    else:
        return TRAINERS[cfg.main.retrieval_strat](device, cfg)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DictConfig):
    force_cudnn_initialization()

    main_utils.validate_cfg(cfg)

    WORLD_SIZE = torch.cuda.device_count()

    main_utils.ddp_setup(RANK, WORLD_SIZE)

    if cfg.main.eval_batch is not None:
        max_batch_sizes[cfg.main.model_name]["eval"] = cfg.main.eval_batch

    # reproducibility stuff
    try:
        torch.manual_seed(cfg.main.ds_seed + RANK)
        torch.cuda.manual_seed_all(cfg.main.ds_seed + RANK)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    model = model_utils.init_model(cfg.main.model_name)

    if cfg.main.bf16:
        model = model.to(torch.bfloat16)

    main_utils.print_rank_0(f"Params: {sum([p.numel() for p in model.parameters()])}")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(RANK)

    main_utils.print_rank_0("Initialized model")

    # init model
    model = DDP(model, device_ids=[RANK], find_unused_parameters=True)

    setting = data_utils.Settings[cfg.main.setting.replace(" ", "_").upper()]

    # set up ordered list of datasets
    dataset_list = data_utils.dataset_configs[cfg.main.datasets] if cfg.main.datasets in data_utils.dataset_configs else [cfg.main.datasets]
    proxy_datasets = proxy_dataset_configs[cfg.main.datasets] if cfg.main.datasets in proxy_dataset_configs else []

    if len(dataset_list) == 1:
        assert cfg.main.datasets == "debug" or cfg.main.ft_upper_bound, f"{cfg.main.datasets} {cfg.main.ft_upper_bound}"

    # ensure order of datasets goes according to the ds seed
    ds_rng = np.random.default_rng(seed=cfg.main.ds_seed)
    if not cfg.main.test_only:
        ds_rng.shuffle(dataset_list)

    main_utils.print_rank_0(f"Seed: {cfg.main.ds_seed} Order: {dataset_list}")

    # trainers train the model batch-by-batch and can implement continual learning strategies along the way
    trainer = init_trainer(torch.device(RANK), cfg)
    
    # save the best model so far here so we can use the correct model when moving onto the next dataset + revert after early stopping
    best_model_pth = os.path.join("tmp/best_model", f"{socket.gethostname()}.pth")
    pre_ds_model_pth = os.path.join("tmp/pre_ds_model", f"{socket.gethostname()}.pth")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.main.swil_ada == "ds" and cfg.main.retrieval_strat == "SWIL":
        no_SWIL_datasets = SWIL_DATASET_MAP[cfg.main.swil_ds_config]

    with wandb.init(project=cfg.main.wandb_proj, mode=cfg.main.wandb_mode if RANK == 0 else "disabled", config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)):
        """
        Training loop
        """
        for dataset_i, current_dataset_name in enumerate(dataset_list):
            if cfg.main.swil_ada == "ds" and cfg.main.retrieval_strat == "SWIL":
                if current_dataset_name not in no_SWIL_datasets:
                    trainer.SWIL_active = True
                else:
                    trainer.SWIL_active = False

            if current_dataset_name in no_FT_datasets:
                main_utils.print_rank_0(f"Skipping {current_dataset_name}")
                continue

            if cfg.main.test_only == True:
                main_utils.print_rank_0("Testing only!")
                break

            # get a dataset-specific config object
            with open(os.path.join("configs/ds_specific/sweeps", cfg.main.sweep_name, f"{current_dataset_name}.pkl"), "rb") as f:
                all_hp_sets = pickle.load(f)[cfg.main.model_name]

                if cfg.main.override_epochs is not None:
                    for hp_set in all_hp_sets:
                        hp_set["epochs"] = cfg.main.override_epochs

            # may only be using the first HP set in the list
            if cfg.main.use_first_set_only:
                all_hp_sets = all_hp_sets[:1]

            val_loader, _ = data_utils.get_loader(
                split="val",
                batch_size=get_batch_size(cfg.main.model_name, "eval", cfg),
                dataset_name=current_dataset_name,
                setting=setting,
                TDS_model_id=cfg.main.model_name,
                big_ds_HDF5=cfg.main.retrieval_strat is not None
            )

            # save the model as it is before training on the current dataset
            if RANK == 0:
                torch.save(model.module.state_dict(), pre_ds_model_pth)

            if len(all_hp_sets) > 1:
                assert cfg.main.retrieval_strat is None, "Must first ensure all loop logic (like dedup)"

            best_val_mAP_ds = -1 # best val accuracy for all HP sets (for the dataset/ds)
            best_HP_set = None # best HP set dict
            no_improvement_streak = 0
            for hp_set_i, hp_set in enumerate(all_hp_sets):
                best_val_mAP_hp_set = -1 # best val accuracy for the current HP set

                # some datasets have custom batch sizes since they are so smol
                train_batch_size = get_batch_size(cfg.main.model_name, "train", cfg) if hp_set["train_batch"] is None else hp_set["train_batch"]

                # may be using a replay-concatenated dataset on following epochs
                train_loader, ds_len = data_utils.get_loader(
                    split="train",
                    batch_size=train_batch_size if cfg.main.retrieval_strat is None else max([1, train_batch_size // 2]),
                    dataset_name=current_dataset_name,
                    setting=setting,
                    TDS_model_id=cfg.main.model_name,
                    n_ways=cfg.main.n_ways,
                    k_shots=cfg.main.k_shots,
                    big_ds_HDF5=cfg.main.retrieval_strat is not None,
                    subset_id=hp_set["subset_id"]
                )

                # set up layerwise LR decay. can just set the multiplier to 1.0 to have no decay
                params = model_utils.apply_lw_lr_decay(
                    model, 
                    cfg.main.model_name, 
                    lrs=[ss_lr * train_batch_size * WORLD_SIZE for ss_lr in hp_set["ss_lrs"]], 
                    lr_mults=hp_set["lw_lr_decay_mults"]
                )

                # instantiate optimizer. will set the learning rates of the two towers below
                if cfg.main.optim == "AdamW":
                    main_utils.print_rank_0("Using AdamW")
                    optimizer = torch.optim.AdamW(params, lr=0, weight_decay=cfg.main.weight_decay)
                elif cfg.main.optim == "SGD":
                    main_utils.print_rank_0("Using SGD")
                    optimizer = torch.optim.SGD(params, lr=0)
                else:
                    raise Exception(f"Unrecognized optimizer: {cfg.main.optim}")

                # instantiate lr scheduler
                if cfg.main.retrieval_strat is None:
                    warmup_steps = hp_set["warmup_epochs"] * len(train_loader)
                    total_steps = hp_set["epochs"] * len(train_loader)
                else:
                    warmup_steps = len(train_loader) + (hp_set["warmup_epochs"] - 1) * len(train_loader) * 2
                    total_steps = len(train_loader) + (hp_set["epochs"] - 1) * len(train_loader) * 2

                if hp_set["lr_sched"] == "cos warm":
                    lr_scheduler = main_utils.WarmupCosineAnnealingScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
                elif hp_set["lr_sched"] == "const warm":
                    lr_scheduler = main_utils.WarmupConstantScheduler(optimizer, warmup_steps=warmup_steps)
                elif hp_set["lr_sched"] is not None:
                    raise Exception(f"Unrecognized LR scheduler: {cfg.main.lr_sched}")
                else:
                    lr_scheduler = None

                # open ds_info since we'll need it at test-time
                with open(f"data/ds_info/{current_dataset_name}.pkl", "rb") as f:
                    ds_info_dict = pickle.load(f)

                # training loop
                torch.cuda.empty_cache() 
                dist.barrier()
                start = time.time()
                for epoch in range(hp_set["epochs"]):
                    max_lr = np.max([p["lr"] for p in optimizer.param_groups])
                    main_utils.wandb_log_rank_0({f"maxlr_{hp_set_i}_{current_dataset_name}": max_lr})
                    
                    if epoch == 1 and not cfg.main.retrieval_each_batch and cfg.main.retrieval_strat is not None: 
                        # if performing replay get the same loader but with replay exemplars concatenated to the dataset
                        train_loader, _ = data_utils.get_loader(
                            split="train",
                            batch_size=train_batch_size,
                            dataset_name=current_dataset_name,
                            setting=setting,
                            TDS_model_id=cfg.main.model_name,
                            n_ways=cfg.main.n_ways,
                            k_shots=cfg.main.k_shots,
                            return_replay_loader=True,
                            replay_buffer=trainer.replay_buffer.as_dataset(ds_len),
                            big_ds_HDF5=cfg.main.retrieval_strat is not None
                        )

                    train_loader.sampler.set_epoch(epoch)

                    progress_str = f"({dataset_i+1}/{len(dataset_list)}, {hp_set_i+1}/{len(all_hp_sets)}, {epoch+1}/{hp_set['epochs']})"
                    main_utils.print_rank_0("------------", progress_str)
                    main_utils.wandb_log_rank_0({"progress": progress_str})

                    model.module.precompute_box_bias(train_batch_size)

                    num_batches = len(train_loader)
                    for batch_i, batch in enumerate(train_loader):
                        inputs, y = main_utils.prepare_batch(batch, cfg.main.prompt_type, torch.device(RANK))

                        trainer.train_batch(inputs=inputs, y=y, model=model, optimizer=optimizer, epoch=epoch)
        
                        if batch_i % 10 == 0 and batch_i != 0:
                            progress_str = f"{batch_i / num_batches * 100:.2f}%"
                            main_utils.print_rank_0(progress_str)
                            main_utils.wandb_log_rank_0({"epoch_progress": progress_str})

                        if lr_scheduler is not None:
                            lr_scheduler.step()

                    trainer.post_epoch_logs()
                    model.module.precompute_box_bias(get_batch_size(cfg.main.model_name, "eval", cfg))

                    torch.cuda.empty_cache()

                    # do not validate if setting is OFSL or if this is the final dataset in the sequence
                    if current_dataset_name != "master":
                        main_utils.print_rank_0(f"Validating on {hp_set_i} {current_dataset_name} after epoch {epoch}")

                        metrics = main_utils.test(
                            val_loader, 
                            model=model,
                            ds_info=ds_info_dict,
                            split="val",
                            lvis_eval="LVIS" in current_dataset_name.upper(),
                            k_preds_per_image=k_preds_per_image(current_dataset_name)
                        )

                        epoch_mAP = main_utils.get_mAP(metrics)

                        main_utils.print_rank_0(f"--------- val_{hp_set_i}_{current_dataset_name}_mAP: {epoch_mAP}")
                        main_utils.wandb_log_rank_0({f"val_{hp_set_i}_{current_dataset_name}_mAP": epoch_mAP})

                        # perform early stopping if necessary
                        if cfg.main.early_stopping:
                            if best_val_mAP_hp_set - epoch_mAP > cfg.main.early_stopping_thresh:
                                if no_improvement_streak > cfg.main.early_stopping_patience + 1:
                                    main_utils.wandb_log_rank_0({f"{hp_set_i}_{current_dataset_name}_epochs": epoch + 1})
                                    break
                                else:
                                    no_improvement_streak += 1
                            else:
                                no_improvement_streak = 0

                        if epoch_mAP > best_val_mAP_ds:
                            # save the best model for this dataset such that we can continue training on the next dataset
                            if RANK == 0:
                                torch.save(model.module.state_dict(), best_model_pth)
                            best_val_mAP_ds = epoch_mAP
                            best_HP_set = copy.deepcopy(hp_set)
                        
                        if epoch_mAP > best_val_mAP_hp_set:
                            # just in case we early-stop on a future epoch, reload the best checkpoint as to minimize the number of epochs trained
                            best_val_mAP_hp_set = epoch_mAP

                    if epoch == hp_set["epochs"] - 1:
                        main_utils.wandb_log_rank_0({f"{hp_set_i}_{current_dataset_name}_epochs": epoch + 1})

                    if cfg.main.retrieval_strat is not None and cfg.main.reset_dedup_on == "epoch":
                        trainer.reset_dedup()

                    torch.cuda.empty_cache()

                """
                On to the next HP set; load the model from before training on this dataset and log best accuracy for this HP set
                """
                if not cfg.main.use_first_set_only:
                    main_utils.print_rank_0(f'--------- bval_{hp_set_i}_{current_dataset_name}_mAP: {best_val_mAP_hp_set}')
                    main_utils.wandb_log_rank_0({f'bval_{hp_set_i}_{current_dataset_name}_mAP': best_val_mAP_hp_set})

                # load the model 
                torch.cuda.empty_cache()
                dist.barrier()
                model.module.load_state_dict(torch.load(pre_ds_model_pth))
                
                if cfg.main.bf16:
                    model = model.to(torch.bfloat16)

                if cfg.main.retrieval_strat is not None and cfg.main.reset_dedup_on == "dataset":
                    trainer.reset_dedup()

                dist.barrier()

            """
            On to the next dataset; load the best model (from all HP sets) and log best accuracy for this dataset
            """
            main_utils.print_rank_0(current_dataset_name, best_HP_set)

            torch.cuda.empty_cache()
            dist.barrier()

            model.module.load_state_dict(torch.load(best_model_pth))  
            if cfg.main.bf16:
                model = model.to(torch.bfloat16)

            if cfg.main.save_ds_models:
                main_utils.save_model(
                    model.module,
                    cfg.main.wandb_proj,
                    wandb.run.name,
                    cfg.main.datasets,
                    cfg.main.model_name,
                    ds_id=f"{current_dataset_name}_{dataset_i}"
                )

            main_utils.print_rank_0(f'--------- bval_{current_dataset_name}_mAP: {best_val_mAP_ds}')
            main_utils.wandb_log_rank_0({f'bval_{current_dataset_name}_mAP': best_val_mAP_ds})
            
            del train_loader
            del val_loader
            gc.collect()

            # may be proxy testing after each dataset
            if cfg.main.proxy_each_dataset:
                split = "test" if cfg.main.testing else "val"
                for current_proxy_name in proxy_datasets:
                    proxy_loader, _ = data_utils.get_loader(
                        split=split,
                        batch_size=get_batch_size(cfg.main.model_name, "eval", cfg),
                        dataset_name=current_proxy_name,
                        setting=setting,
                        TDS_model_id=cfg.main.model_name,
                        big_ds_HDF5=cfg.main.retrieval_strat is not None
                    )

                    with open(f"data/ds_info/{current_proxy_name}.pkl", "rb") as f:
                        proxy_ds_info_dict = pickle.load(f)

                    if bool(cfg.main.testing):
                        main_utils.print_rank_0(f"Running proxy test on {current_proxy_name}")
                    else:
                        main_utils.print_rank_0(f"Running proxy validation on {current_proxy_name}")

                    metrics = main_utils.test(
                        proxy_loader, 
                        model=model,
                        ds_info=proxy_ds_info_dict,
                        split=split,
                        lvis_eval="LVIS" in current_proxy_name.upper(),
                        k_preds_per_image=k_preds_per_image(current_proxy_name),
                        cls_specific=cfg.main.cls_specific
                    )

                    performance = main_utils.get_mAP(metrics, "LVIS" in current_proxy_name.upper())

                    if isinstance(performance, dict):
                        for key, val in performance.items():
                            main_utils.print_rank_0(f"--------- {split}_{current_proxy_name}_{current_dataset_name}_{dataset_i}_{key}_mAP: {val}")
                            main_utils.wandb_log_rank_0({f"{split}_{current_proxy_name}_{current_dataset_name}_{dataset_i}_{key}_mAP": val})
                    else:
                        main_utils.print_rank_0(f"--------- {split}_{current_proxy_name}_{current_dataset_name}_{dataset_i}_mAP: {performance}")
                        main_utils.wandb_log_rank_0({f"{split}_{current_proxy_name}_{current_dataset_name}_{dataset_i}_mAP": performance})
                
                    del proxy_loader
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()

        """
            Training is done; test on all test sets or all validation sets
        """
        if cfg.main.save_model and RANK == 0:
            main_utils.save_model(
                model.module,
                cfg.main.wandb_proj,
                wandb.run.name,
                cfg.main.datasets,
                cfg.main.model_name
            )

        if cfg.main.chkpt_path is not None and cfg.main.test_only:
            print(f"Loading {cfg.main.chkpt_path}")
            model.module.load_state_dict(torch.load(cfg.main.chkpt_path))

        model.module.precompute_box_bias(get_batch_size(cfg.main.model_name, "eval", cfg))

        torch.cuda.empty_cache()
        dist.barrier()

        split = "test" if bool(cfg.main.testing) else "val"
        all_perfs = []
        for current_dataset_name in dataset_list:
            if cfg.main.proxy_test_only or current_dataset_name in no_test_datasets:
                continue
            main_utils.print_rank_0(f"Running final {split} on {current_dataset_name}")

            with open(f"data/ds_info/{current_dataset_name}.pkl", "rb") as f:
                ds_info_dict = pickle.load(f)

            loader, _ = data_utils.get_loader(
                split=split,
                batch_size=get_batch_size(cfg.main.model_name, "eval", cfg),
                dataset_name=current_dataset_name,
                setting=setting,
                TDS_model_id=cfg.main.model_name,
                big_ds_HDF5=cfg.main.retrieval_strat is not None
            )

            start = time.time()
            metrics = main_utils.test(
                loader, 
                model=model,
                ds_info=ds_info_dict,
                split=split,
                lvis_eval="LVIS" in current_dataset_name.upper(),
                k_preds_per_image=k_preds_per_image(current_dataset_name)
            )
            performance = main_utils.get_mAP(metrics, "LVIS" in current_dataset_name.upper())

            if isinstance(performance, dict):
                for key, val in performance.items():
                    main_utils.print_rank_0(f"--------- f{split}_{current_dataset_name}_{key}_mAP: {val}")
                    main_utils.wandb_log_rank_0({f"f{split}_{current_dataset_name}_{key}_mAP": val})
            else:
                main_utils.print_rank_0(f"--------- f{split}_{current_dataset_name}_mAP: {performance}")
                main_utils.wandb_log_rank_0({f"f{split}_{current_dataset_name}_mAP": performance})
                all_perfs.append(performance)

            main_utils.print_rank_0(f"Took {time.time() - start:.2f}s to test")

            del loader
            gc.collect()
        
        main_utils.wandb_log_rank_0({f"ODinW_mAP": np.mean(all_perfs)})

        # finally, test on the proxy datasets
        if cfg.main.ft_upper_bound:
            main_utils.print_rank_0("Skipping proxy testing")

        for current_dataset_name in proxy_datasets:
            if cfg.main.ft_upper_bound:
                continue

            split = "test" if cfg.main.testing else "val"

            loader, _ = data_utils.get_loader(
                split=split,
                batch_size=get_batch_size(cfg.main.model_name, "eval", cfg),
                dataset_name=current_dataset_name,
                setting=setting,
                TDS_model_id=cfg.main.model_name,
                big_ds_HDF5=cfg.main.retrieval_strat is not None
            )

            with open(f"data/ds_info/{current_dataset_name}.pkl", "rb") as f:
                ds_info_dict = pickle.load(f)

            if bool(cfg.main.testing):
                main_utils.print_rank_0(f"Running proxy test on {current_dataset_name}")
            else:
                main_utils.print_rank_0(f"Running proxy validation on {current_dataset_name}")

            metrics = main_utils.test(
                loader, 
                model=model,
                ds_info=ds_info_dict,
                split=split,
                lvis_eval="LVIS" in current_dataset_name.upper(),
                k_preds_per_image=k_preds_per_image(current_dataset_name),
                cls_specific=cfg.main.cls_specific
            )

            performance = main_utils.get_mAP(metrics, "LVIS" in current_dataset_name.upper())

            if isinstance(performance, dict):
                for key, val in performance.items():
                    main_utils.print_rank_0(f"--------- f{split}_{current_dataset_name}_{key}_mAP: {val}")
                    main_utils.wandb_log_rank_0({f"f{split}_{current_dataset_name}_{key}_mAP": val})
            else:
                main_utils.print_rank_0(f"--------- f{split}_{current_dataset_name}_mAP: {performance}")
                main_utils.wandb_log_rank_0({f"f{split}_{current_dataset_name}_mAP": performance})
        
            del loader
            gc.collect()

        if cfg.main.save_model and RANK == 0:
            main_utils.save_model(
                model.module,
                cfg.main.wandb_proj,
                wandb.run.name,
                cfg.main.datasets,
                cfg.main.model_name
            )

    destroy_process_group()
        
        
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    mp.set_start_method('spawn')
    mp.spawn(launch_main, nprocs=torch.cuda.device_count(), join=True)


"""
Tools for logging memory usage

if args["log_mem"]:
    columns = ['Obj Type', 'Size', "is_cuda"]
    data = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                data.append([str(type(obj)), str(obj.size()), str(obj.is_cuda)])
            elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                data.append([str(type(obj)), str(obj.data.size()), str(obj.data.is_cuda)])
        except:
            pass
    table = wandb.Table(data=data, columns=columns)
    main_utils.wandb_log_rank_0({"memory_pre_update": table})
"""