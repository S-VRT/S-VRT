# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ----------------------------------
# Environment variable reading with fallbacks
# ----------------------------------
def get_local_rank():
    """Get local rank from environment variables.
    Priority: LOCAL_RANK > SLURM_LOCALID > 0
    """
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    return 0


def get_rank():
    """Get global rank from environment variables.
    Priority: RANK > SLURM_PROCID > distributed.get_rank() > 0
    """
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size from environment variables.
    Priority: WORLD_SIZE > SLURM_NTASKS > distributed.get_world_size() > 1
    """
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    elif dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def is_dist_available_and_initialized():
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


# ----------------------------------
# init
# ----------------------------------
def setup_distributed():
    """
    Modern distributed training setup supporting both platform DDP and torchrun.
    
    Platform DDP: Platform injects RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* env vars
                  and runs the same command for each process.
    
    Local torchrun: User runs `torchrun --nproc_per_node=N script.py`
    
    This function auto-detects the environment and initializes accordingly.
    Only initializes distributed if WORLD_SIZE > 1.
    """
    world_size = get_world_size()
    
    if world_size <= 1:
        # Single process mode - no distributed initialization needed
        return
    
    # Get rank and local_rank
    rank = get_rank()
    local_rank = get_local_rank()
    
    # Set device BEFORE initializing process group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize process group using env:// (reads MASTER_ADDR, MASTER_PORT from env)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # Verify initialization
    if dist.is_initialized():
        assert dist.get_rank() == rank, f"Rank mismatch: env={rank}, dist={dist.get_rank()}"
        assert dist.get_world_size() == world_size, f"World size mismatch: env={world_size}, dist={dist.get_world_size()}"


def init_dist(launcher, backend='nccl', **kwargs):
    """Legacy function for backward compatibility. Prefer setup_distributed()."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    """Legacy pytorch launcher initialization. Use setup_distributed() instead."""
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)



# ----------------------------------
# get rank and world_size (legacy compatibility)
# ----------------------------------
def get_dist_info():
    """Legacy function. Returns (rank, world_size) tuple."""
    return get_rank(), get_world_size()


def master_only(func):
    """Decorator to run function only on rank 0."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


# ----------------------------------
# Synchronization utilities
# ----------------------------------
def barrier():
    """
    Synchronization barrier across all processes.
    Only executes if distributed training is initialized.
    """
    if is_dist_available_and_initialized():
        dist.barrier()


def all_reduce_mean(tensor):
    """
    All-reduce a tensor and compute mean across all ranks.
    
    Args:
        tensor: PyTorch tensor to reduce
        
    Returns:
        Averaged tensor (same shape as input)
    """
    if not is_dist_available_and_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    # Clone to avoid modifying original
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)
    
    return tensor






# ----------------------------------
# operation across ranks
# ----------------------------------
def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def gather_grad(params):
    world_size = get_world_size()
    
    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def all_gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    local_size = torch.IntTensor([tensor.numel()]).to('cuda')
    size_list = [torch.IntTensor([0]).to('cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to('cuda'))

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to('cuda')
        tensor = torch.cat((tensor, padding), 0)

    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses


def barrier_safe():
    """
    安全的barrier函数，只在分布式模式下执行同步
    
    用于确保所有进程在关键点（如保存checkpoint）保持同步，
    避免进程间状态不一致导致的DDP同步错误。
    
    在非分布式模式下调用此函数不会有任何操作。
    
    Note: This is an alias for barrier() for backward compatibility.
    """
    barrier()
