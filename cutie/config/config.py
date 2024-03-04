from hydra import compose, initialize

initialize(version_base='1.3.2', config_path="", job_name="gui")
global_config = compose(config_name="gui_config")