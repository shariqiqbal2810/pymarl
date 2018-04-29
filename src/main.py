import numpy as np
import pymongo
from sacred import Experiment, SETTINGS
from sacred.arg_parser import parse_args
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from components.transforms import _merge_dicts
from run import run
from utils.logging import get_logger

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("deepmarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# The central mongodb for our deepmarl experiments
# You need to set up local port forwarding to ensure this local port maps to the server
db_host = "localhost"
db_port = 27027 # Use a different port from the default mongodb port to avoid a potential clash
client = None
mongodb_fail = False
if True:
    # First try to connect to the central server. If that doesn't work then just save locally
    maxSevSelDelay = 100  # Assume 1ms maximum server selection delay
    try:
        # Check whether server is accessible
        logger.info("Trying to connect to mongoDB '{}:{}'".format(db_host, db_port))
        client = pymongo.MongoClient(db_host, db_port, serverSelectionTimeoutMS=maxSevSelDelay)
        client.server_info()
        # If this hasn't raised an exception, we can add the observer
        ex.observers.append(MongoObserver.create(url=db_host, port=db_port, db_name='deepmarl'))
        logger.info("Added MongoDB observer on {}.".format(db_host))
    except pymongo.errors.ServerSelectionTimeoutError:
        logger.warning("Couldn't connect to MongoDB.")
        logger.info("Fallback to FileStorageObserver in ./results/sacred.")
        mongodb_fail = True
if mongodb_fail:
    import os
    from os.path import dirname, abspath
    file_obs_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
    logger.info("Using the FileStorageObserver in ./results/sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

@ex.main
def my_main(_run, _config, _log, env_args):

    # Setting the random seed throughout the modules
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    env_args['seed'] = _config["seed"]

    # run the framework
    run(_run, _config, _log, client)


if __name__ == '__main__':
    from copy import deepcopy
    params = deepcopy(sys.argv)

    defaults = []
    config_dic = {}

    # manually parse for experiment tags
    del_indices = []
    exp_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--exp_name":
            del_indices.append(_i)
            exp_name = _v.split("=")[1]
            break

    # load experiment config (if there is such as thing)
    exp_dic = None
    if exp_name is not None:
        from config.experiments import REGISTRY as exp_REGISTRY
        assert exp_name in exp_REGISTRY, "Unknown experiment name: {}".format(exp_name)
        exp_dic = exp_REGISTRY[exp_name](None, logger)
        if "defaults" in exp_dic:
            defaults.extend(exp_dic["defaults"].split(" "))
            del exp_dic["defaults"]
        config_dic = deepcopy(exp_dic)

    # check for defaults in command line parameters
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--default_cfgs":
            del_indices.append(_i)
            defaults.extend(_v.split("=")[1].split(" "))
            break

    # load default configs in order
    for _d in defaults:
        from config.defaults import REGISTRY as def_REGISTRY
        def_dic = def_REGISTRY[_d](config_dic, logger)
        config_dic = _merge_dicts(config_dic, def_dic)

    #  finally merge with experiment config
    if exp_name is not None:
        config_dic = _merge_dicts(config_dic, exp_dic)

    # now add all the config to sacred
    ex.add_config(config_dic)

    # delete indices that contain custom experiment tags
    for _i in sorted(del_indices, reverse=True):
        del params[_i]

    #_setup_mongodb(params)
    ex.run_commandline(params)

