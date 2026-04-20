import sys
import datetime
import logging
import os
import numpy as np

# Optional dependencies for TensorBoard, WANDB and SwanLab
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except Exception:
    swanlab = None
    SWANLAB_AVAILABLE = False

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except Exception:
    logfire = None
    LOGFIRE_AVAILABLE = False


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


class _LogfireLoggingHandler(logging.Handler):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record):
        if getattr(record, '_svrt_skip_logfire', False):
            return
        if not self.bridge.enabled or not self.bridge.text_enabled:
            return
        try:
            level_name = record.levelname.lower()
            log_method = getattr(self.bridge.logfire, level_name, self.bridge.logfire.info)
            event_message = record.getMessage()
            log_method(
                event_message,
                message=event_message,
                logger_name=record.name,
                level=record.levelname,
                pathname=record.pathname,
                lineno=record.lineno,
                log_origin=getattr(record, 'log_origin', 'train_core'),
                launch_stream=getattr(record, 'launch_stream', None),
                launch_phase=getattr(record, 'launch_phase', None),
                launch_mode=getattr(record, 'launch_mode', None),
                launch_command=getattr(record, 'launch_command', None),
                **self.bridge.context,
            )
        except Exception as e:
            self.bridge._disable_channel('text', e)


def logger_info(logger_name, log_path='default_logger.log', opt=None, add_stream_handler=True, verbose=True):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    Enhanced to create timestamped log files for each run
    '''
    # Avoid attaching handlers on non-master processes to prevent duplicate logs
    try:
        rank_env = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    except Exception:
        rank_env = 0
    if rank_env != 0:
        # Return early for non-master ranks; they should not create handlers
        return

    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        if verbose:
            print('LogHandlers exist!')
    else:
        if verbose:
            print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        
        # Generate timestamp for log file
        timestamp = datetime.datetime.now().strftime('_%y%m%d_%H%M%S')

        # Process log_path to add timestamp
        if os.path.isfile(log_path):
            # Caller passed an existing file — use it directly (append mode).
            log_file = log_path
        elif os.path.isdir(log_path):
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

        # Use 'a' mode if the file already exists (e.g. wrapper subprocesses
        # appending to a log created by ensure_launch_logger), otherwise 'w'.
        file_mode = 'a' if os.path.exists(log_file) else 'w'
        fh = logging.FileHandler(log_file, mode=file_mode)
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        action = 'appending to' if file_mode == 'a' else 'created'
        if verbose:
            print(f'Log file {action}: {log_file}')

        if add_stream_handler:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            log.addHandler(sh)

    if opt is not None:
        bridge = getattr(log, '_svrt_logfire_bridge', None)
        if bridge is None:
            bridge = _LogfireBridge(opt, logger=log)
            log._svrt_logfire_bridge = bridge

        existing_logfire_handlers = [
            handler for handler in log.handlers if isinstance(handler, _LogfireLoggingHandler)
        ]
        if bridge.enabled and bridge.text_enabled and not existing_logfire_handlers:
            log.addHandler(_LogfireLoggingHandler(bridge))


def emit_launch_wrapper_log(
    logger_name,
    level,
    message,
    log_origin='launch_wrapper',
    launch_stream=None,
    launch_phase=None,
    launch_mode=None,
    launch_command=None,
):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        raise RuntimeError(
            f'Logger "{logger_name}" is not initialized. Call logger_info() first.'
        )

    extra = {
        'log_origin': log_origin,
        'launch_stream': launch_stream,
        'launch_phase': launch_phase,
        'launch_mode': launch_mode,
        'launch_command': launch_command,
    }

    effective_level = str(level).lower()
    valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
    if effective_level not in valid_levels:
        raise ValueError(
            f'Invalid log level "{effective_level}". Must be one of {sorted(valid_levels)}.'
        )
    log_method = getattr(logger, effective_level)
    log_method(message, extra=extra)


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


def _compact_logfire_fields(fields):
    return {k: v for k, v in fields.items() if v is not None}


class _LogfireBridge:
    def __init__(self, opt, logger=None):
        self.logger = logger
        self.logfire = None
        self.enabled = False
        self.text_enabled = False
        self.metrics_enabled = False
        self.timings_enabled = False
        self._disabled_channels = set()
        self._warned_messages = set()

        logging_config = opt.get('logging', {})
        rank = opt.get('rank', 0)
        if rank != 0:
            self.context = {}
            return

        if not logging_config.get('use_logfire', False):
            self.context = {}
            return

        project_name = logging_config.get('logfire_project_name')
        service_name = logging_config.get('logfire_service_name', 's-vrt')
        environment = logging_config.get('logfire_environment')
        run_name = (
            logging_config.get('wandb_name')
            or logging_config.get('swanlab_name')
            or opt.get('task')
        )

        self.context = _compact_logfire_fields(
            {
                'task': opt.get('task'),
                'opt_path': opt.get('opt_path'),
                'rank': rank,
                'world_size': opt.get('world_size'),
                'is_train': opt.get('is_train'),
                'project_name': project_name,
                'service_name': service_name,
                'environment': environment,
                'run_name': run_name,
            }
        )

        if not LOGFIRE_AVAILABLE or logfire is None:
            self._warn_once('Logfire is not available. Please install logfire: pip install logfire')
            return

        configure_kwargs = _compact_logfire_fields(
            {
                'token': logging_config.get('logfire_token'),
                'service_name': service_name,
                'environment': environment,
            }
        )

        try:
            logfire.configure(**configure_kwargs)
        except Exception as e:
            self._warn_once(f'Failed to initialize Logfire: {e}')
            return

        self.logfire = logfire
        self.enabled = True
        self.text_enabled = logging_config.get('logfire_log_text', True)
        self.metrics_enabled = logging_config.get('logfire_log_metrics', True)
        self.timings_enabled = logging_config.get('logfire_log_timings', True)

    def _warn_once(self, message):
        if message in self._warned_messages:
            return
        self._warned_messages.add(message)
        if self.logger:
            self.logger.warning(message, extra={'_svrt_skip_logfire': True})
        else:
            print(f'Warning: {message}')

    def _disable_channel(self, channel, exc):
        if channel in self._disabled_channels:
            return
        self._disabled_channels.add(channel)
        if channel == 'text':
            self.text_enabled = False
        elif channel == 'metrics':
            self.metrics_enabled = False
        elif channel == 'timings':
            self.timings_enabled = False
        self._warn_once(f'Disabling Logfire {channel} logging after error: {exc}')

    def emit_metrics(self, step, scalar_dict, tag_prefix=''):
        if not (self.enabled and self.metrics_enabled and self.logfire is not None):
            return
        if not scalar_dict:
            return

        prefix = tag_prefix.rstrip('/') + '/' if tag_prefix else ''
        metrics = {
            f'{prefix}{key}': value
            for key, value in scalar_dict.items()
            if value is not None
        }
        if not metrics:
            return

        try:
            self.logfire.info('svrt metrics', step=step, metrics=metrics, **self.context)
        except Exception as e:
            self._disable_channel('metrics', e)

    def emit_timings(self, step, timings_dict, prefix='timings'):
        if not (self.enabled and self.timings_enabled and self.logfire is not None):
            return
        if not timings_dict:
            return

        timings = {
            f'{prefix}/{key}': value
            for key, value in timings_dict.items()
            if value is not None
        }
        if not timings:
            return

        try:
            self.logfire.info('svrt timings', step=step, timings=timings, **self.context)
        except Exception as e:
            self._disable_channel('timings', e)


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
    
    def __init__(self, opt, logger=None, timer=None):
        self.opt = opt
        self.logger = logger
        self.tb_writer = None
        self.wandb_run = None
        self.swanlab_run = None
        # Optional Timer instance can be injected for logging timings
        self.timer = timer
        
        # Get logging configuration
        logging_config = opt.get('logging', {})
        attached_bridge = getattr(logger, '_svrt_logfire_bridge', None) if logger is not None else None
        self.logfire_bridge = attached_bridge if attached_bridge is not None else _LogfireBridge(opt, logger=logger)
        if logger is not None and attached_bridge is None:
            setattr(logger, '_svrt_logfire_bridge', self.logfire_bridge)
        self.use_tensorboard = logging_config.get('use_tensorboard', False)
        self.use_wandb = logging_config.get('use_wandb', False)
        self.use_swanlab = logging_config.get('use_swanlab', False)
        self.swanlab_auto_resume = logging_config.get('swanlab_auto_resume', True)
        self.swanlab_resume_strategy = logging_config.get('swanlab_resume_strategy', 'allow')
        self.swanlab_manual_run_id = logging_config.get('swanlab_run_id', None)
        self.swanlab_run_id_file = self._init_swanlab_run_id_file(
            logging_config.get('swanlab_run_id_file', None),
            opt
        )
        
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
                    wandb_mode = logging_config.get('wandb_mode', None)
                    if not wandb_mode:
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

        # Initialize SwanLab
        if self.use_swanlab:
            if not SWANLAB_AVAILABLE:
                if self.logger:
                    self.logger.warning('SwanLab is not available. Please install swanlab: pip install swanlab')
                else:
                    print('Warning: SwanLab is not available. Please install swanlab: pip install swanlab')
                self.use_swanlab = False
            else:
                swanlab_api_key = logging_config.get('swanlab_api_key', None)
                if swanlab_api_key:
                    os.environ['SWANLAB_API_KEY'] = swanlab_api_key
                    if self.logger:
                        self.logger.info('SwanLab API key set from configuration (non-interactive mode)')
                    else:
                        print('SwanLab API key set from configuration (non-interactive mode)')
                elif 'SWANLAB_API_KEY' not in os.environ:
                    warning_msg = ('SwanLab API key not found in config or environment. '
                                   'Please set swanlab_api_key in config or run: swanlab login')
                    if self.logger:
                        self.logger.warning(warning_msg)
                    else:
                        print(f'Warning: {warning_msg}')

                swanlab_project = logging_config.get('swanlab_project', opt.get('task', 'experiment'))
                swanlab_workspace = logging_config.get('swanlab_workspace', None)
                swanlab_name = logging_config.get('swanlab_name', opt.get('task', 'experiment'))
                swanlab_description = logging_config.get('swanlab_description', None)
                swanlab_mode = logging_config.get('swanlab_mode', None)

                if not swanlab_mode:
                    swanlab_mode = 'cloud' if (swanlab_api_key or 'SWANLAB_API_KEY' in os.environ) else 'offline'

                swanlab_kwargs = {'config': opt}
                if swanlab_project:
                    swanlab_kwargs['project'] = swanlab_project
                if swanlab_workspace:
                    swanlab_kwargs['workspace'] = swanlab_workspace
                if swanlab_name:
                    swanlab_kwargs['experiment_name'] = swanlab_name
                if swanlab_description:
                    swanlab_kwargs['description'] = swanlab_description
                if swanlab_mode:
                    swanlab_kwargs['mode'] = swanlab_mode

                swanlab_resume_id = self.swanlab_manual_run_id
                if swanlab_mode == 'cloud' and swanlab_resume_id is None and self.swanlab_auto_resume:
                    swanlab_resume_id = self._load_swanlab_run_id()

                resume_strategy = self.swanlab_resume_strategy if swanlab_mode == 'cloud' else None
                if resume_strategy:
                    swanlab_kwargs['resume'] = resume_strategy

                if swanlab_mode == 'cloud' and swanlab_resume_id:
                    swanlab_kwargs['id'] = swanlab_resume_id
                    if self.logger:
                        self.logger.info(f'Resuming SwanLab run with id={swanlab_resume_id}')
                    else:
                        print(f'Resuming SwanLab run with id={swanlab_resume_id}')

                try:
                    self.swanlab_run = swanlab.init(**swanlab_kwargs)
                    if self.logger:
                        self.logger.info(
                            f'SwanLab logging enabled. Project: {swanlab_project}, '
                            f'Name: {swanlab_name}, Mode: {swanlab_mode}'
                        )
                    else:
                        print(f'SwanLab logging enabled. Project: {swanlab_project}, '
                              f'Name: {swanlab_name}, Mode: {swanlab_mode}')
                    if self.swanlab_auto_resume:
                        self._persist_swanlab_run_id(getattr(self.swanlab_run, 'id', None))
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'Failed to initialize SwanLab: {e}')
                    else:
                        print(f'Warning: Failed to initialize SwanLab: {e}')
                    self.use_swanlab = False
    
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

        # Log to SwanLab
        if self.use_swanlab and self.swanlab_run is not None:
            swanlab_dict = {f'{tag_prefix}{key}': value for key, value in scalar_dict.items() if value is not None}
            if swanlab_dict:
                try:
                    self.swanlab_run.log(swanlab_dict, step=step)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'Failed to log scalars to SwanLab: {e}')
                    else:
                        print(f'Warning: Failed to log scalars to SwanLab: {e}')
                    self.use_swanlab = False

        self.logfire_bridge.emit_metrics(step, scalar_dict, tag_prefix=tag_prefix)
    
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

        # Log to SwanLab
        if self.use_swanlab and self.swanlab_run is not None:
            swanlab_dict = {}
            for key, image in image_dict.items():
                if image is None:
                    continue
                swanlab_image = self._to_swanlab_image(image)
                if swanlab_image is not None:
                    swanlab_dict[f'{tag_prefix}{key}'] = swanlab_image
            if swanlab_dict:
                try:
                    self.swanlab_run.log(swanlab_dict, step=step)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f'Failed to log images to SwanLab: {e}')
                    else:
                        print(f'Warning: Failed to log images to SwanLab: {e}')
                    self.use_swanlab = False
    
    def log_timings(self, step=None, timings_dict=None, prefix='timings'):
        """
        Log timing information. Accepts either a timings dict or uses the injected Timer.
        timings_dict: dict[str, float] mapping name->seconds
        """
        # resolve timings source
        if timings_dict is None:
            timings_source = getattr(self, 'timer', None)
            if timings_source is None:
                return
            # Timer exposes get_current_timings()
            timings_dict = timings_source.get_current_timings()

        if not timings_dict:
            return

        # Format message
        try:
            msg = prefix + ': ' + ' '.join(f'{k}={v:.4f}s' for k, v in timings_dict.items())
        except Exception:
            msg = prefix + ': (unprintable timings)'

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

        # send to TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            for k, v in timings_dict.items():
                try:
                    self.tb_writer.add_scalar(f'{prefix}/{k}', v, step if step is not None else 0)
                except Exception:
                    pass

        # send to WANDB
        if self.use_wandb and self.wandb_run is not None:
            try:
                wandb_dict = {f'{prefix}/{k}': v for k, v in timings_dict.items()}
                if step is not None:
                    wandb_dict['step'] = step
                self.wandb_run.log(wandb_dict, step=step)
            except Exception:
                pass

        self.logfire_bridge.emit_timings(step=step, timings_dict=timings_dict, prefix=prefix)
    
    def log_config(self, config_dict):
        """
        Log configuration to WANDB.
        
        Args:
            config_dict (dict): Configuration dictionary
        """
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.config.update(config_dict, allow_val_change=True)

        if self.use_swanlab and self.swanlab_run is not None:
            try:
                self.swanlab_run.config.update(config_dict)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f'Failed to update SwanLab config: {e}')
                else:
                    print(f'Warning: Failed to update SwanLab config: {e}')
    
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

        if self.swanlab_run is not None:
            self.swanlab_run.finish()
            if self.logger:
                self.logger.info('SwanLab run finished.')

    def _init_swanlab_run_id_file(self, configured_path, opt):
        """
        Compute the path used to persist SwanLab run IDs for auto-resume.
        """
        if not self.swanlab_auto_resume:
            return None

        base_log_dir = opt['path'].get('log', opt['path'].get('task', '.'))
        run_id_path = configured_path if configured_path is not None else os.path.join(base_log_dir, 'swanlab_run.id')

        if not os.path.isabs(run_id_path):
            run_id_path = os.path.join(base_log_dir, run_id_path)

        os.makedirs(os.path.dirname(run_id_path), exist_ok=True)
        return run_id_path

    def _load_swanlab_run_id(self):
        """
        Load the cached SwanLab run ID if available.
        """
        if not self.swanlab_run_id_file or not os.path.isfile(self.swanlab_run_id_file):
            return None
        try:
            with open(self.swanlab_run_id_file, 'r') as fp:
                run_id = fp.read().strip()
                return run_id or None
        except OSError as e:
            if self.logger:
                self.logger.warning(f'Failed to read SwanLab run id file: {e}')
            else:
                print(f'Warning: Failed to read SwanLab run id file: {e}')
            return None

    def _persist_swanlab_run_id(self, run_id):
        """
        Persist the SwanLab run ID for future auto-resume attempts.
        """
        if not self.swanlab_run_id_file or not run_id:
            return
        try:
            with open(self.swanlab_run_id_file, 'w') as fp:
                fp.write(run_id)
        except OSError as e:
            if self.logger:
                self.logger.warning(f'Failed to write SwanLab run id file: {e}')
            else:
                print(f'Warning: Failed to write SwanLab run id file: {e}')

    def _to_swanlab_image(self, image):
        """
        Prepare numpy image data for SwanLab logging.
        """
        if not SWANLAB_AVAILABLE:
            return None

        np_image = self._to_numpy_image(image)
        if np_image is None:
            return None

        try:
            return swanlab.Image(np_image)
        except Exception as e:
            if self.logger:
                self.logger.warning(f'Failed to convert image for SwanLab logging: {e}')
            else:
                print(f'Warning: Failed to convert image for SwanLab logging: {e}')
            return None

    @staticmethod
    def _to_numpy_image(image):
        """
        Convert tensor/array image to HWC numpy format for logging.
        """
        if image is None:
            return None
        try:
            if hasattr(image, 'detach'):
                image = image.detach()
            if hasattr(image, 'cpu'):
                image = image.cpu()
            if hasattr(image, 'numpy'):
                image = image.numpy()
        except Exception:
            pass

        if isinstance(image, np.ndarray):
            np_image = image
        else:
            return None

        if np_image.ndim == 3 and np_image.shape[0] in (1, 3):
            np_image = np.transpose(np_image, (1, 2, 0))
        elif np_image.ndim == 2:
            np_image = np_image[:, :, None]

        return np.clip(np_image, 0.0, 1.0)
