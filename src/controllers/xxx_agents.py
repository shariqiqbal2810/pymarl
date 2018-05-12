import numpy as np
from torch.autograd import Variable
import torch as th

from components.action_selectors import REGISTRY as as_REGISTRY
from components import REGISTRY as co_REGISTRY
from components.scheme import Scheme
from components.episode_buffer import BatchEpisodeBuffer
from components.transforms import _build_model_inputs, _join_dicts, _generate_scheme_shapes, _generate_input_shapes
from itertools import combinations
from models import REGISTRY as mo_REGISTRY

class XXXMultiagentController():
    """
    container object for a set of independent agents
    TODO: may need to propagate test_mode in here as well!
    """

    def __init__(self, runner, n_agents, n_actions, action_selector=None, args=None):
        self.args = args
        self.runner = runner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agent_str = args.agent
        assert self.args.agent_output_type in ["policies"], "agent_output_type has to be set to 'policies' for XXX - makes no sense with other methods!"
        self.agent_output_type = "policies"

        self.model_level1 = mo_REGISTRY[args.xxx_agent_model_level1]
        self.model_level2 = mo_REGISTRY[args.xxx_agent_model_level2]
        self.model_level3 = mo_REGISTRY[args.xxx_agent_model_level3]

        # # Set up action selector
        if action_selector is None:
            self.action_selector = as_REGISTRY[args.action_selector](args=self.args)
        else:
            self.action_selector = action_selector

        self.agent_scheme_level1 = Scheme([dict(name="observations",
                                                select_agent_ids=list(range(self.n_agents))),
                                           dict(name="actions_level1",
                                                rename="past_action_level1",
                                                transforms=[("shift", dict(steps=1)),
                                                            ("one_hot", dict(range=(0, self.n_actions-1)))],
                                                switch=self.args.xxx_obs_last_actions_level1),
                                           dict(name="xxx_epsilon_central_level1",
                                                scope="episode"),
                                           dict(name="xxx_epsilon_level1")
                                           ])


        self.agent_scheme_level2_fn = lambda _agent_id1, _agent_id2: Scheme([dict(name="agent_id",
                                                                                  rename="agent_ids",
                                                                                  transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                                  select_agent_ids=[_agent_id1, _agent_id2],),
                                                                             dict(name="observations",
                                                                                  select_agent_ids=[_agent_id1, _agent_id2]),
                                                                             *[dict(name="actions_level2_agents{}:{}".format(_agent_id1, _agent_id2),
                                                                                    rename="past_actions_level2_agents{}:{}".format(_agent_id1, _agent_id2),
                                                                                    transforms=[("shift", dict(steps=1)),
                                                                                                ("one_hot", dict(range=(0, self.n_actions-1)))],
                                                                                    switch=self.args.xxx_obs_last_actions_level2) for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2))],
                                                                             dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id]),
                                                                             dict(name="xxx_epsilon_central_level2",
                                                                                  scope="episode"),
                                                                             dict(name="xxx_epsilon_level2")
                                                                             ])

        self.agent_scheme_level3_fn = lambda _agent_id: Scheme([dict(name="agent_id",
                                                                     transforms=[("one_hot",dict(range=(0, self.n_agents-1)))],
                                                                     select_agent_ids=[_agent_id],),
                                                                dict(name="observations",
                                                                     select_agent_ids=[_agent_id]),
                                                                dict(name="actions_level3",
                                                                     rename="past_actions_level3",
                                                                     select_agent_ids=[_agent_id],
                                                                     transforms=[("shift", dict(steps=1)),
                                                                                 ("one_hot", dict(range=(0, self.n_actions-1)))], # DEBUG!
                                                                     switch=self.args.xxx_obs_last_actions_level3),
                                                                dict(name="agent_id", rename="agent_id__flat", select_agent_ids=[_agent_id]),
                                                                dict(name="xxx_epsilon_central_level3", scope="episode"),
                                                                dict(name="xxx_epsilon_level3")
                                                               ])

        # Set up schemes
        self.schemes = {}
        # level 1
        self.schemes_level1 = {}
        self.schemes_level1["agent_input_level1"] = self.agent_scheme_level1

        # level 2
        self.schemes_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.schemes_level2["agent_input_level2__agents{}:{}".format(_agent_id1, _agent_id2)] = self.agent_scheme_level2_fn(_agent_id1,
                                                                                                                         _agent_id2).agent_flatten()
        # level 3
        self.schemes_level3 = {}
        for _agent_id in range(self.n_agents):
            self.schemes_level3["agent_input__agent{}".format(_agent_id)] = self.agent_scheme_level3_fn(_agent_id).agent_flatten()

        # create joint scheme from the agents schemes
        self.joint_scheme_dict_level1 = _join_dicts(self.schemes_level1)
        self.joint_scheme_dict_level2 = _join_dicts(self.schemes_level2)
        self.joint_scheme_dict_level3 = _join_dicts(self.schemes_level3)

        self.joint_scheme_dict = _join_dicts(self.schemes_level1, self.schemes_level2, self.schemes_level3)
        # construct model-specific input regions

        # level 1
        self.input_columns_level1 = {}
        self.input_columns_level1["agent_input_level1"] = {}
        self.input_columns_level1["agent_input_level1"]["main"] = \
            Scheme([dict(name="observations", select_agent_ids=list(range(self.n_agents))),
                    dict(name="past_actions_level1",
                         switch=self.args.xxx_obs_last_actions_level1),
                    ])
        self.input_columns_level1["agent_input_level1"]["epsilon_central_level1"] = \
            Scheme([dict(name="xxx_epsilon_central_level1")])
        self.input_columns_level1["agent_input_level1"]["epsilon_level1"] = \
            Scheme([dict(name="xxx_epsilon_level1")])

        # level 2
        self.input_columns_level2 = {}
        for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
            self.input_columns_level2["agent_input_level2__agents{}:{}".format(_agent_id1, _agent_id2)] = {}
            self.input_columns_level2["agent_input_level2__agents{}:{}".format(_agent_id1, _agent_id2)]["main"] = \
                Scheme([dict(name="observations", select_agent_ids=[_agent_id1, _agent_id2]),
                        dict(name="past_actions_level2_agents{}:{}".format(_agent_id1, _agent_id2),
                             switch=self.args.xxx_obs_last_actions_level2),
                        dict(name="agent_ids", select_agent_ids=[_agent_id1, _agent_id2])])
            self.input_columns_level2["agent_input_level2__agents{}:{}".format(_agent_id1, _agent_id2)]["epsilon_central_level2"] = \
                Scheme([dict(name="xxx_epsilon_central_level2")])
            self.input_columns_level2["agent_input_level2__agents{}:{}".format(_agent_id1, _agent_id2)]["epsilon_level2"] = \
                Scheme([dict(name="xxx_epsilon_level2")])

        # level 3
        self.input_columns_level3 = {}
        for _agent_id in range(self.n_agents):
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)] = {}
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["main"] = \
                Scheme([dict(name="agent_observation", select_agent_ids=[_agent_id]),
                        dict(name="past_action_level3",
                             select_agent_ids=[_agent_id],
                             switch=self.args.xxx_obs_last_actions_level3),
                        dict(name="agent_id", select_agent_ids=[_agent_id])])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilon_central_level3"] = \
                Scheme([dict(name="xxx_epsilon_central_level3")])
            self.input_columns_level3["agent_input_level3__agent{}".format(_agent_id)]["epsilon_level3"] = \
                Scheme([dict(name="xxx_epsilon_level3")])

        pass

    def get_parameters(self, level):
        if level == 1:
            return list(self.models["level1"].parameters())
        elif level == 2:
            param_list = []
            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                param_list.extend(self.models["level2_{}:{}".format(_agent_id1, _agent_id2)].parameters())
            return param_list
        elif level == 3:
            param_list = []
            for _agent_id in range(self.n_agents):
                param_list.extend(self.models["level3_{}".format(_agent_id)].parameters())
            return param_list

    def select_actions(self, inputs, avail_actions, tformat, info, test_mode=False):
        #selected_actions, modified_inputs, selected_actions_format = \
        #    self.action_selector.select_action(inputs,
        #                                       avail_actions=avail_actions,
        #                                       tformat=tformat,
        #                                       test_mode=test_mode)
        # assert False, "TODO" # TODO
        #return selected_actions, modified_inputs, selected_actions_format

        selected_actions_dict = {}
        selected_actions_dict["actions_level1"] = self.actions_level1
        selected_actions_dict["actions_level2"] = self.actions_level2

        modified_inputs_dict = {}
        modified_inputs_dict["policies_level1"] = self.policies_level1
        modified_inputs_dict["policies_level2"] = self.policies_level2

        return selected_actions_dict, modified_inputs_dict, self.selected_actions_format

    def create_model(self, transition_scheme):

        self.scheme_shapes_level1 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                            dict_of_schemes=self.schemes_level1)

        self.input_shapes_level1 = _generate_input_shapes(input_columns=self.input_columns_level1,
                                                   scheme_shapes=self.scheme_shapes_level1)

        self.scheme_shapes_level2 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.schemes_level2)

        self.input_shapes_level2 = _generate_input_shapes(input_columns=self.input_columns_level2,
                                                   scheme_shapes=self.scheme_shapes_level2)

        self.scheme_shapes_level3 = _generate_scheme_shapes(transition_scheme=transition_scheme,
                                                     dict_of_schemes=self.schemes_level3)

        self.input_shapes_level3 = _generate_input_shapes(input_columns=self.input_columns_level3,
                                                   scheme_shapes=self.scheme_shapes_level3)


        # TODO: Set up agent models
        self.models = {}

        # set up models level 1
        self.models["level1"] = self.model_level1(input_shapes=self.input_shapes_level1["main"],
                                                  n_actions=self.n_actions,
                                                  output_type=self.agent_output_type,
                                                  args=self.args)
        if self.args.use_cuda:
            self.models["level1"] = self.models["level1"].cuda()

        # set up models level 2
        if self.args.share_params:
            model_level2 = self.model_level2(input_shapes=self.input_shapes_level2["main"],
                                             n_actions=self.n_actions,
                                             output_type=self.agent_output_type,
                                             args=self.args)
            if self.args.use_cuda:
                model_level2 = model_level2.cuda()

            for _agent_id1, _agent_id2 in sorted(combinations(list(range(self.n_agents)), 2)):
                self.models["level2_{}:{}".format(_agent_id1, _agent_id2)] = model_level2
        else:
            assert False, "TODO"
        if self.args.use_cuda:
            self.model = self.model.cuda()

        # set up models level 3
        self.models_level3 = {}
        if self.args.share_params:
            model_level3 = self.model_level3(input_shapes=self.input_shapes_level3["main"],
                                             n_actions=self.n_actions,
                                             output_type=self.agent_output_type,
                                             args=self.args)
            if self.args.use_cuda:
                model_level3 = model_level3.cuda()

            for _agent_id in range(self.n_agents):
                self.models["level3_{}".format(_agent_id)] = model_level3
        else:
            assert False, "TODO"

        return

    def generate_initial_hidden_states(self, batch_size):
        """
        generates initial hidden states for each agent
        """

        # Set up hidden states for all levels - and propagate througn the runner!
        hidden_dict = {}
        hidden_dict["level1"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(1)])
        hidden_dict["level2"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(len(sorted(combinations(list(range(self.n_agents)), 2))))])
        hidden_dict["level3"] = th.stack([Variable(th.zeros(batch_size, 1, self.args.agents_hidden_state_size)) for _
                                        in range(self.n_agents)])
        return hidden_dict, "?*bs*v*t"

    def share_memory(self):
        assert False, "TODO"
        pass

    def get_outputs(self, inputs, hidden_states, tformat, loss_fn=None, **kwargs):

        avail_actions = kwargs["avail_actions"]
        test_mode = kwargs["test_mode"]

        if self.args.share_agent_params:
            # TODO: Need to do this over 3 levels
            #
            # inputs, inputs_tformat = _build_model_inputs(self.input_columns,
            #                                              inputs,
            #                                              to_variable=True,
            #                                              inputs_tformat=tformat)
            #
            # out, hidden_states, losses, tformat = self.lambda_network_model(inputs,
            #                                                                 hidden_states=hidden_states,
            #                                                                 loss_fn=loss_fn,
            #                                                                 tformat=inputs_tformat,
            #                                                                 **kwargs)
            # ret = {"hidden_states": hidden_states,
            #        "losses": losses,
            #        "format": tformat}
            #
            # out_key = self.agent_output_type
            # ret[out_key] = out

            # top level: aa' ~ Pi_c sample which pair to coordinate
            # second level: pick up correct pair (given aa'), sample u^a, u^a' from the pair coordinator
            # either decode u^a, u^a' from the sampled action, or refer to level

            # --------------------- LEVEL 1
            inputs_level1, inputs_level1_tformat = _build_model_inputs(self.input_columns_level1,
                                                         inputs,
                                                         to_variable=True,
                                                         inputs_tformat=tformat)

            out_level1, hidden_states_level1, losses_level1, tformat_level1 = self.models["level1"](inputs,
                                                                                                    hidden_states=hidden_states["level1"],
                                                                                                    loss_fn=loss_fn,
                                                                                                    tformat=inputs_level1_tformat,
                                                                                                    **kwargs)

            sampled_pair_ids, modified_inputs, selected_actions_format = self.action_selector.select_action(out_level1,
                                                                                                            avail_actions=None,
                                                                                                            tformat=tformat,
                                                                                                            test_mode=test_mode)

            # sample which pairs should be selected
            self.actions_level1 = sampled_pair_ids.clone()
            self.selected_actions_format = selected_actions_format
            self.policies_level1 = out_level1.clone()

            # --------------------- LEVEL 2
            assert self.n_agents == 3, "pair selection only implemented for 3 agents yet!!"
            tmp = (avail_actions * avail_actions)
            pairwise_avail_actions = th.bmm(tmp.unsqueeze(2), tmp.unsqueeze(1))

            inputs_level2, inputs_level2_tformat = _build_model_inputs(self.input_columns_level2,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       avail_actions=pairwise_avail_actions,
                                                                       sampled_pair_ids=sampled_pair_ids)

            out_level2, hidden_states_level2, losses_level1, tformat_level2 = self.models["level1"](inputs_level2,
                                                                                                    hidden_states=hidden_states["level2"],
                                                                                                    loss_fn=loss_fn,
                                                                                                    tformat=inputs_level2_tformat,
                                                                                                    **kwargs)

            pair_sampled_actions, \
            modified_inputs, \
            selected_actions_format_level2 = self.action_selector.select_action(out_level2,
                                                                         avail_actions=None,
                                                                         tformat=tformat,
                                                                         test_mode=test_mode)

            self.actions_level2 = pair_sampled_actions.clone()
            self.selected_actions_format_level2 = selected_actions_format
            self.policies_level2 = out_level2.clone()

            # --------------------- LEVEL 3
            agent_ids_not_sampled_yet = # TODO

            inputs_level3, inputs_level3_tformat = _build_model_inputs(self.input_columns_level3,
                                                                       inputs,
                                                                       to_variable=True,
                                                                       inputs_tformat=tformat,
                                                                       #avail_actions=pairwise_avail_actions,
                                                                       #sampled_pair_ids=sampled_pair_ids,
                                                                       agent_ids_not_sampled_yet=agent_ids_not_sampled_yet)
            self.actions_level3 = pair_sampled_actions.clone()
            self.selected_actions_format_level3 = selected_actions_format
            self.policies_level3 = out_level3.clone()

            return #ret, tformat
        else:
            assert False, "Not yet implemented."

    def save_models(self, path, token, T):
        if self.args.share_agent_params:
            th.save(self.agent_model.state_dict(),
                    "results/models/{}/{}_agentsp__{}_T.weights".format(token, self.args.learner, T))
        else:
            for _agent_id in range(self.args.n_agents):
                th.save(self.agent_models["agent__agent{}".format(_agent_id)].state_dict(),
                        "results/models/{}/{}_agent{}__{}_T.weights".format(token, self.args.learner, _agent_id, T))
        pass

    pass




