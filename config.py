from enum import Enum

class ConfigName(Enum):
    RAND_SEED = "RAND_SEED"

_CONFIG = {
    # Add default configuration parameters
}

def get_config(key=None, default=None):
    if key is None:
        return _CONFIG
    return _CONFIG.get(key, default)

def set_config(key, value):
    _CONFIG[key] = value