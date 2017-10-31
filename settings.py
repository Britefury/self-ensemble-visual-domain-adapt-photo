import sys, os
if sys.version_info[0] == 2:
    from ConfigParser import RawConfigParser
else:
    from configparser import RawConfigParser

_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        if os.path.exists('datasets.cfg'):
            _CONFIG = RawConfigParser()
            _CONFIG.read('datasets.cfg')
        else:
            raise ValueError('Could not find configuration file datasets.cfg')
    return _CONFIG


def get_data_dir(name):
    config = get_config()
    path = config.get('paths', name)
    if path is not None and path != '':
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise ValueError('Configuration file entry for paths:{} does not exist'.format(name))
        return path
    else:
        raise ValueError('Configuration file did not have entry for paths:{}'.format(name))
