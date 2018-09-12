REGISTRY = {}

from .q_learner import QLearner
from .icql_learner import ICQLLearner
from .coma_learner import COMALearner
from .actor_critic_learner import ActorCriticLearner

REGISTRY["q_learner"] = QLearner
REGISTRY["icql_learner"] = ICQLLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner