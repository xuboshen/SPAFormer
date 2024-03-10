import torch
def build_model(cfg, model, gpu_id=None):
    if torch.cuda.is_available():
        assert (
            cfg.num_gpus <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.num_gpus == 0
        ), "Cuda is not available. Please set `num_gpus: 0 for running on CPUs."

    if cfg.num_gpus:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.num_gpus > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
    return model
