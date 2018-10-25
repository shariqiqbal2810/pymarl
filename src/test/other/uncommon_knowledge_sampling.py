
from torch.distributions import Categorical

probs = [[0.1, 0.1], [0.8, 0.8]]

pi_1 = Categorical([p for i in range(len(probs)) for p in probs[i, :]])

for t in range(100000):
    x = 0