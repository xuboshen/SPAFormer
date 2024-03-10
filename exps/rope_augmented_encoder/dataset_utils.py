from torch.utils.data.distributed import DistributedSampler
def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=cfg.seed) if cfg.num_gpus > 1 else None

    return sampler