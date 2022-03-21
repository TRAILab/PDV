import os

import wandb


def is_wandb_enabled(cfg):
    wandb_enabled = False
    if cfg.get('WANDB'):
        wandb_enabled = cfg.WANDB.ENABLED
    if cfg.get('LOCAL_RANK'):
        wandb_enabled &= cfg.LOCAL_RANK == 0
    # TODO Check WANDB_API_KEY validity, try except?
    if os.environ.get('WANDB_API_KEY') is None:
        wandb_enabled = False
    return wandb_enabled


def init(cfg, args, job_type='train_eval'):
    """
        Initialize wandb by passing in config
    """
    if not is_wandb_enabled(cfg):
        return

    wandb.init(name=cfg.TAG,
               config=cfg,
               project=cfg.WANDB.PROJECT,
               entity=cfg.WANDB.ENTITY,
               tags=[args.extra_tag],
               job_type=job_type)


def log(cfg, log_dict, step):
    if not is_wandb_enabled(cfg):
        return

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)

    wandb.log(log_dict, step)


def summary(cfg, log_dict, step, highest_metric=-1):
    """
    Wandb summary information
    Args:
        cfg
    """

    if not is_wandb_enabled(cfg):
        return

    assert isinstance(log_dict, dict)
    assert isinstance(step, int)

    metric = log_dict.get(cfg.WANDB.get('SUMMARY_HIGHEST_METRIC'))
    if metric is not None and metric > highest_metric:
        # wandb overwrites summary with last epoch run. Append '_best' to keep highest metric
        for key, value in log_dict.items():
            wandb.run.summary[key + '_best'] = value
        wandb.run.summary['epoch'] = step
        highest_metric = metric

    return highest_metric


def log_and_summary(cfg, log_dict, step, highest_metric=-1):
    log(cfg, log_dict, step)
    return summary(cfg, log_dict, step, highest_metric)
