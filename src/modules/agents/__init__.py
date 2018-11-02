REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .ff_agent import FFAgent
REGISTRY["ff"] = FFAgent
