import sys
import datetime
import logging
import os

# Optional dependencies for TensorBoard and WANDB
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# --------------------------------------------
# logger
# --------------------------------------------
'''


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    Enhanced to create timestamped log files for each run
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        
        # Generate timestamp for log file
        timestamp = datetime.datetime.now().strftime('_%y%m%d_%H%M%S')
        
        # Process log_path to add timestamp
        if os.path.isdir(log_path):
            # If log_path is a directory, create log file with timestamp
            log_file = os.path.join(log_path, logger_name + timestamp + '.log')
        else:
            # If log_path is a file path, insert timestamp before extension
            dir_name = os.path.dirname(log_path)
            file_name = os.path.basename(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            name, ext = os.path.splitext(file_name)
            log_file = os.path.join(dir_name, name + timestamp + ext) if dir_name else (name + timestamp + ext)
        
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Use 'w' mode to create a new file for each run instead of appending
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        print(f'Log file created: {log_file}')

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
'''


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


'''
# --------------------------------------------
# Unified Logger for File, TensorBoard and WANDB
# --------------------------------------------
'''


class Logger(object):
    """
    Unified logger that manages file logging, TensorBoard, and WANDB.
    
    Args:
        opt (dict): Configuration dictionary containing logging settings
        logger (logging.Logger, optional): Python logger for file logging
    """
    
    def __init__(self, opt, logger=None):
        self.opt = opt
        self.logger = logger
        self.tb_writer = None
        self.wandb_run = None
        
        # Get logging configuration
        logging_config = opt.get('logging', {})
        self.use_tensorboard = logging_config.get('use_tensorboard', False)
        self.use_wandb = logging_config.get('use_wandb', False)
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                if self.logger:
                    self.logger.warning('TensorBoard is not available. Please install tensorboard: pip install tensorboard')
                else:
                    print('Warning: TensorBoard is not available. Please install tensorboard: pip install tensorboard')
                self.use_tensorboard = False
            else:
                tb_log_dir = opt['path'].get('tensorboard', os.path.join(opt['path']['log'], 'tensorboard'))
                os.makedirs(tb_log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
                if self.logger:
                    self.logger.info(f'TensorBoard logging enabled at: {tb_log_dir}')
                else:
                    print(f'TensorBoard logging enabled at: {tb_log_dir}')
        
        # Initialize WANDB
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                if self.logger:
                    self.logger.warning('WANDB is not available. Please install wandb: pip install wandb')
                else:
                    print('Warning: WANDB is not available. Please install wandb: pip install wandb')
                self.use_wandb = False
            else:
                # Set WANDB API key from config for non-interactive login
                wandb_api_key = logging_config.get('wandb_api_key', None)
                if wandb_api_key:
                    os.environ['WANDB_API_KEY'] = wandb_api_key
                    if self.logger:
                        self.logger.info('WANDB API key set from configuration (non-interactive mode)')
                    else:
                        print('WANDB API key set from configuration (non-interactive mode)')
                elif 'WANDB_API_KEY' not in os.environ:
                    if self.logger:
                        self.logger.warning('WANDB API key not found in config or environment. '
                                         'Please set wandb_api_key in config or run: wandb login')
                    else:
                        print('Warning: WANDB API key not found in config or environment. '
                              'Please set wandb_api_key in config or run: wandb login')
                
                wandb_project = logging_config.get('wandb_project', 'kair-training')
                wandb_entity = logging_config.get('wandb_entity', None)
                wandb_name = logging_config.get('wandb_name', opt.get('task', 'experiment'))
                
                try:
                    # Set mode to offline if no API key is available
                    wandb_mode = 'online' if (wandb_api_key or 'WANDB_API_KEY' in os.environ) else 'offline'
                    
                    self.wandb_run = wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=wandb_name,
                        config=opt,
                        resume='allow',
                        mode=wandb_mode
                    )
                    if self.logger:
                        self.logger.info(f'WANDB logging enabled. Project: {wandb_project}, Name: {wandb_name}, Mode: {wandb_mode}')
                    else:
                        print(f'WANDB logging enabled. Project: {wandb_project}, Name: {wandb_name}, Mode: {wandb_mode}')
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'Failed to initialize WANDB: {e}')
                    else:
                        print(f'Warning: Failed to initialize WANDB: {e}')
                    self.use_wandb = False
    
    def log_scalars(self, step, scalar_dict, tag_prefix=''):
        """
        Log scalar values to TensorBoard and WANDB.
        
        Args:
            step (int): Training step/iteration
            scalar_dict (dict): Dictionary of scalar names and values
            tag_prefix (str): Prefix for the tags (e.g., 'train', 'test')
        """
        if not scalar_dict:
            return
        
        # Add prefix to tags if provided
        if tag_prefix:
            tag_prefix = tag_prefix.rstrip('/') + '/'
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in scalar_dict.items():
                if value is not None:
                    self.tb_writer.add_scalar(f'{tag_prefix}{key}', value, step)
        
        # Log to WANDB
        if self.use_wandb and self.wandb_run is not None:
            wandb_dict = {f'{tag_prefix}{key}': value for key, value in scalar_dict.items() if value is not None}
            wandb_dict['step'] = step
            self.wandb_run.log(wandb_dict, step=step)
    
    def log_images(self, step, image_dict, tag_prefix=''):
        """
        Log images to TensorBoard and WANDB.
        
        Args:
            step (int): Training step/iteration
            image_dict (dict): Dictionary of image names and tensors/arrays
            tag_prefix (str): Prefix for the tags (e.g., 'train', 'test')
        """
        if not image_dict:
            return
        
        # Add prefix to tags if provided
        if tag_prefix:
            tag_prefix = tag_prefix.rstrip('/') + '/'
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            for key, image in image_dict.items():
                if image is not None:
                    # Assume image is in CHW or HWC format
                    self.tb_writer.add_image(f'{tag_prefix}{key}', image, step, dataformats='CHW')
        
        # Log to WANDB
        if self.use_wandb and self.wandb_run is not None:
            wandb_dict = {}
            for key, image in image_dict.items():
                if image is not None:
                    # Convert tensor to numpy if needed
                    if hasattr(image, 'cpu'):
                        image = image.cpu().numpy()
                    wandb_dict[f'{tag_prefix}{key}'] = wandb.Image(image)
            if wandb_dict:
                wandb_dict['step'] = step
                self.wandb_run.log(wandb_dict, step=step)
    
    def log_config(self, config_dict):
        """
        Log configuration to WANDB.
        
        Args:
            config_dict (dict): Configuration dictionary
        """
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.config.update(config_dict, allow_val_change=True)
    
    def close(self):
        """
        Close all logging writers.
        """
        if self.tb_writer is not None:
            self.tb_writer.close()
            if self.logger:
                self.logger.info('TensorBoard writer closed.')
        
        if self.wandb_run is not None:
            self.wandb_run.finish()
            if self.logger:
                self.logger.info('WANDB run finished.')
