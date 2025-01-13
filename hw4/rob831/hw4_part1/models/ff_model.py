from torch import nn
import torch
from torch import optim
from rob831.hw4_part1.models.base_model import BaseModel
from rob831.hw4_part1.infrastructure.utils import normalize, unnormalize
from rob831.hw4_part1.infrastructure import pytorch_util as ptu


class FFModel(nn.Module, BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        # super(FFModel, self).__init__()
        super().__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        :param obs_unnormalized: Unnormalized observations
        :param acs_unnormalized: Unnormalized actions
        :param obs_mean: Mean of observations
        :param obs_std: Standard deviation of observations
        :param acs_mean: Mean of actions
        :param acs_std: Standard deviation of actions
        :param delta_mean: Mean of state difference `s_t+1 - s_t`.
        :param delta_std: Standard deviation of state difference `s_t+1 - s_t`.
        :return: tuple `(next_obs_pred, delta_pred_normalized)`
        This forward function should return a tuple of two items
            1. `next_obs_pred` which is the predicted `s_t+1`
            2. `delta_pred_normalized` which is the normalized (i.e. not
                unnormalized) output of the delta network. This is needed
        """
        # Check if obs_unnormalized is a tensor and move it to the CPU if necessary
        # if isinstance(obs_unnormalized, torch.Tensor) and obs_unnormalized.is_cuda:
        #     obs_unnormalized = ptu.to_numpy(obs_unnormalized)

        # if isinstance(acs_unnormalized, torch.Tensor) and acs_unnormalized.is_cuda:
        #     acs_unnormalized = ptu.to_numpy(acs_unnormalized)

        # delta_mean = self.data_statistics['delta_mean']
        # delta_std = self.data_statistics['delta_std']
        # normalize input data to mean 0, std 1
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)# TODO(Q1)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)# TODO(Q1)


        # check that 2d, if not add dim
        if len(obs_normalized.shape) == 1:
            obs_normalized = obs_normalized.unsqueeze(0)  
        if len(acs_normalized.shape) == 1:
            acs_normalized = acs_normalized.unsqueeze(0)  

        # predicted change in obs
        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1) compute delta_pred_normalized and next_obs_pred
        # Hint: as described in the PDF, the output of the network is the
        # *normalized change* in state, i.e. normalized(s_t+1 - s_t).
        delta_pred_normalized = self.delta_network(concatenated_input)# TODO(Q1)
        # next_obs_pred = obs_normalized + delta_pred_normalized# TODO(Q1)
        next_obs_pred = obs_unnormalized + unnormalize(delta_pred_normalized, delta_mean, delta_std)
        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        :param obs: numpy array of observations (s_t)
        :param acs: numpy array of actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        obs_mean = data_statistics['obs_mean']
        obs_std = data_statistics['obs_std']
        acs_mean = data_statistics['acs_mean']
        acs_std = data_statistics['acs_std']
        delta_mean = data_statistics['delta_mean']
        delta_std = data_statistics['delta_std']

        #make everything a tensor:
        obs_mean = ptu.from_numpy(obs_mean)
        obs_std = ptu.from_numpy(obs_std)
        acs_mean = ptu.from_numpy(acs_mean)
        acs_std = ptu.from_numpy(acs_std)
        delta_mean = ptu.from_numpy(delta_mean)
        delta_std = ptu.from_numpy(delta_std)
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)

        with torch.no_grad():
          pred, _ = self.forward(obs, acs, obs_mean,
              obs_std, acs_mean, acs_std, delta_mean, delta_std)
        # TODO(Q1) get the predicted next-states (s_t+1) as a numpy array
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.

        prediction = ptu.to_numpy(pred)
        return prediction

    def update(self, observations, actions, next_observations, data_statistics):
        """
        :param observations: numpy array of observations
        :param actions: numpy array of actions
        :param next_observations: numpy array of next observations
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return:
        """

        # Hint: you should use `data_statistics['delta_mean']` and
        # `data_statistics['delta_std']`, which keep track of the mean
        # and standard deviation of the model.     
        obs_mean = data_statistics['obs_mean']
        obs_std = data_statistics['obs_std']
        acs_mean = data_statistics['acs_mean']
        acs_std = data_statistics['acs_std']
        delta_mean = data_statistics['delta_mean']
        delta_std = data_statistics['delta_std']

        #make everything a tensor:
        obs_mean = ptu.from_numpy(obs_mean)
        obs_std = ptu.from_numpy(obs_std)
        acs_mean = ptu.from_numpy(acs_mean)
        acs_std = ptu.from_numpy(acs_std)
        delta_mean = ptu.from_numpy(delta_mean)
        delta_std = ptu.from_numpy(delta_std)
        observations = ptu.from_numpy(observations)
        next_observations = ptu.from_numpy(next_observations)
        actions = ptu.from_numpy(actions)

        ground_t_delta = (next_observations - observations)
        ground_t_delta_normalized = normalize(ground_t_delta, delta_mean, delta_std)

        _, delta_pred_normalized = self.forward(observations, actions, obs_mean, obs_std, acs_mean, 
        acs_std, delta_mean, delta_std) # TODO(Q1) compute the normalized target for the model.
        
        # print(f"target type: {type(target)}, input type: {type(next_observations)}")

        loss = self.loss(delta_pred_normalized, ground_t_delta_normalized) # TODO(Q1) compute the loss
        # self.loss(target, ptu.from_numpy(next_observations))
        # Hint: `self(...)` returns a tuple, but you only need to use one of the
        # outputs.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }



        ##### DRAFT ##########################################
        # pred_target, _ = self.forward(observations, actions, obs_mean, obs_std, acs_mean, 
        # acs_std, delta_mean, delta_std)

        # target = pred_target
        # self.get_prediction(observations, actions, data_statistics)

                # # check if not already tensor --> if not, convert to tensor
        # if not isinstance(obs_normalized, torch.Tensor):
        #     obs_normalized = ptu.from_numpy(obs_normalized)
        # if not isinstance(acs_normalized, torch.Tensor):
        #  obs_normalized = ptu.from_numpy(obs_normalized)
        #     acs_normalized = ptu.from_numpy(acs_normalized)

        # if not isinstance(delta_mean, torch.Tensor):
        #   delta_mean = ptu.from_numpy(delta_mean)
        # if not isinstance(delta_std, torch.Tensor):
        #   delta_std = ptu.from_numpy(delta_std)

        # if not isinstance(obs_normalized, torch.Tensor):
        #     obs_normalized = ptu.from_numpy(obs_normalized)
        # if not isinstance(acs_normalized, torch.Tensor):
        #     acs_normalized = ptu.from_numpy(acs_normalized)


        # #make everything a tensor:
        # obs_mean = ptu.from_numpy(obs_mean)
        # obs_std = ptu.from_numpy(obs_mean)
        # acs_mean = ptu.from_numpy(obs_mean)
        # acs_std = ptu.from_numpy(obs_mean)
        # delta_mean = ptu.from_numpy(obs_mean)
        # delta_std = ptu.from_numpy(obs_mean)
