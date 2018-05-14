from itertools import combinations
import torch as th

def _n_agent_pair_samples(n_agents):
    return n_agents // 2

def _ordered_agent_pairings(n_agents):
    return sorted(combinations(list(range(n_agents)), 2))

def _n_agent_pairings(n_agents):
    return int((n_agents * (n_agents-1)) / 2)

def _joint_actions_2_action_pair(joint_action, n_actions, use_delegate_action=True):
    if use_delegate_action:
        mask = (joint_action == 0.0)
        joint_action[mask] = 1.0
        _action1 = th.floor((joint_action-1.0) / n_actions)
        _action2 = (joint_action-1.0) % n_actions
        _action1[mask] = float("nan")
        _action2[mask] = float("nan")
    else:
        _action1 = th.floor(joint_action / n_actions)
        _action2 = (joint_action) % n_actions
    return _action1, _action2

def _action_pair_2_joint_actions(action_pair, n_actions):
    return action_pair[0] * n_actions + action_pair[1]

def _pairing_id_2_agent_ids(pairing_id, n_agents):
    all_pairings = _ordered_agent_pairings(n_agents)
    return all_pairings[pairing_id]

def _pairing_id_2_agent_ids__tensor(pairing_id, n_agents):
    pairing_list = _ordered_agent_pairings(n_agents)
    ttype = th.cuda.LongTensor if pairing_id.is_cuda else th.LongTensor
    all_pairings = ttype(pairing_list).unsqueeze(2).repeat(1,1,pairing_id.shape[0])
    ret0 = all_pairings[:,0,:].gather(0, pairing_id.unsqueeze(0).long())
    ret1 = all_pairings[:,1,:].gather(0, pairing_id.unsqueeze(0).long())
    return ret0, ret1

def _agent_ids_2_pairing_id(agent_ids, n_agents):
    agent_ids = tuple(agent_ids)
    all_pairings = _ordered_agent_pairings(n_agents)
    assert agent_ids in all_pairings, "agent_ids is not of proper format!"
    return all_pairings.index(agent_ids)

# simple tests to establish correctness of encoding functions
if __name__ == "__main__":
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(100, 3), 3))
    print(_action_pair_2_joint_actions(_joint_actions_2_action_pair(5, 3), 3))