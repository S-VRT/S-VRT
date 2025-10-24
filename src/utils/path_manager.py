import datetime
from pathlib import Path
import shutil

def create_experiment_dir(repo_root: Path, config_path: Path) -> Path:
    """
    Create a unique experiment directory based on config name and timestamp.
    
    Args:
        repo_root: Root directory of the repository
        config_path: Path to the configuration file
        
    Returns:
        Path to the created experiment directory
    """
    config_name = config_path.stem
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = repo_root / "outputs" / config_name / run_name
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "configs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visuals").mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot
    snapshot_path = exp_dir / "configs" / config_path.name
    shutil.copy2(config_path, snapshot_path)
    
    return exp_dir

def get_config_from_exp_dir(exp_dir: Path) -> Path:
    """
    Get the config snapshot path from experiment directory.
    Assumes there's exactly one .yaml file in configs/.
    """
    config_dir = exp_dir / "configs"
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config file found in {config_dir}")
    if len(config_files) > 1:
        raise ValueError(f"Multiple config files found in {config_dir}")
    return config_files[0]

def get_checkpoint_path(exp_dir: Path, checkpoint: str = "best.pth") -> Path:
    """
    Get path to a specific checkpoint in the experiment directory.
    
    Args:
        exp_dir: Experiment directory
        checkpoint: Checkpoint filename or 'best'/'last'
        
    Returns:
        Path to the checkpoint file
    """
    ckpt_dir = exp_dir / "checkpoints"
    
    if checkpoint == "best":
        checkpoint = "best.pth"
    elif checkpoint == "last":
        checkpoint = "last.pth"
    
    ckpt_path = ckpt_dir / checkpoint
    if not ckpt_path.exists():
        # Fallback to latest if specified file not found
        ckpt_files = list(ckpt_dir.glob("*.pth"))
        if ckpt_files:
            return max(ckpt_files, key=lambda p: p.stat().st_mtime)
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return ckpt_path

