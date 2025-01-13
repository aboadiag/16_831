import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.uniform(self.low, self.high, size = 
            [num_sequences, horizon, self.ac_dim])
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                pass

            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            cem_action = None

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        # print('cand shape rt', candidate_action_sequences.shape)

        # if obs only has 1 dimension, make (1, obs_dim)
        if len(obs.shape) == 1:
          obs = np.expand_dims(obs, axis = 0)

        total_rewards = np.zeros(candidate_action_sequences.shape[0])

        for model in self.dyn_models: 
            sum_of_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            total_rewards += sum_of_rewards

        mean_prediction = total_rewards/len(self.dyn_models)


        # print('mean pred shape', mean_prediction.shape)

        mean_prediction/= self.horizon  

        # print('mean pred final shape', mean_prediction.shape)

        return mean_prediction

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # idx_best = np.argmax(predicted_rewards) 
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = predicted_rewards.argmax(axis =0)
            # candidate_action_sequences[idx_best] # TODO (Q2)
            # np.argmax(predicted_rewards, axis = 0)#None  
            action_to_take = candidate_action_sequences[best_action_sequence, 0] #--> first actin #None  # TODO (Q2) 
            # print('best_act shape', action_to_take[None].shape)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.

        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`

        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.

        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        count=0 
        sum_of_rewards = np.zeros(self.N)#[] #None  # TODO (Q2)
          
        if len(obs.shape) == 1:  # If single observation
            current_obs = np.expand_dims(obs, axis=0)  # Shape: [1, 4]
        else:
            current_obs = obs  # Shape: [N, 4]
        

        for j in range(self.horizon): 
          # expand obs for first itr 
          if count == 0:
              expanded_obs = np.repeat(current_obs, self.N, axis=0) # [N, 4]
          else:
              expanded_obs = current_obs #now already [N, 4]

          #update count
          count += 1  

          action = candidate_action_sequences[:, j, :] # [N, D_action]

          predicted_obs = model.get_prediction(expanded_obs, action, self.data_statistics)
          # print('pred obs shape', predicted_obs.shape)

          rewards, _ = self.env.get_reward(predicted_obs, action)
          # print('rewards  shape', rewards[0].shape)

          sum_rewards_per_seq = rewards
          # , axis=0)  # Sum over time steps or other dimensions

          sum_of_rewards += sum_rewards_per_seq

          current_obs = predicted_obs
          # print('count', count)



        return sum_of_rewards


################ DRAFT ####################

   # print('seq hmmm', sum_rewards_per_seq[0])
          #update current obs
          # current_obs = predicted_obs
    
        # print('sum of rews shape', sum_of_rewards.shape)
          # expanded_obs = np.repeat(current_obs, self.N, axis=0)  # Shape: [N, D_obs]

          # print('action shape', action.shape)
          # print('current_obs shape', current_obs.shape)
          # print('expanded_obs shape', expanded_obs.shape)
          # if predicted_obs.shape[0] != self.N:
          #   raise ValueError(f"Shape mismatch: predicted_obs has shape {predicted_obs.shape}, expected {self.N}")
        
          # if predicted_obs.shape[0] == 1:
          #   predicted_obs = np.squeeze(predicted_obs, axis=0)  # Remove the first dimension if it's 1

        # assert len(obs.shape) == 2 and obs.shape[1] == self.ob_dim, \
        #   f"Observation should have shape [N, {self.ob_dim}], but got {obs.shape}"
        # assert len(candidate_action_sequences.shape) == 3 and candidate_action_sequences.shape[1] == self.horizon, \
        #   f"Action sequences should have shape [N, H, {self.ac_dim}], but got {candidate_action_sequences.shape}"
            # if len(predicted_rewards.shape) == 1: #if 1d
            #   best_action_sequence = np.argmax(predicted_rewards)
            # else: #if 2d
        # predicted_obs = np.zeros(obs.shape)
        # print('current obs shape', obs.shape)
        #current obs
        # current_obs = obs
              #  if len(action.shape) == 1:
              # action = np.expand_dims(action, axis = 0)
        # print('cand data shape hur', candidate_action_sequences.shape[0])

          # print('shape hur', candidate_action_sequences.shape[0])

          # for i in range(candidate_action_sequences.shape[0]):

            # action = candidate_action_sequences[i]



            # print('action shape', action.shape)
        # mean_prediction = np.zeros(self.N)

                      # print('no models', len(self.dyn_models))

          # for i in range(self.horizon):
              # print('obs shape here', obs.shape)
              # print('cand act shape here', candidate_action_sequences.shape)
              # pass
              # actions = np.squeeze(candidate_action_sequences, axis = 1)
              # print('actions shape hur', actions.shape)
            # model.get_prediction(obs, action, self.data_statistics)
            # pred_sum_rews = sum_of_rewards
        # for candidate_action_sequences[0]:

        # if len(action.shape) == 1:
        #   action = np.expand_dims(action, axis = 0)

        #   #action batch
        #   actions = action
        #   print('acs shape', actions.shape)



        #   sum_of_rewards.append(sum_rewards_per_seq)

          # print('obs shape', obs.shape) obs shape --> 4, --> obs, action, rew, next_obs
          # print('cand act seq', candidate_action_sequences.shape)
          # cand act seq (1000, 10, 2)
          # 1000 act seqs considered
          # 10 is horizon length
          # 2 is action dimension
          # print('act shape', candidate_action_sequences[:, i].shape)


        # for action in candidate_action_sequences:
        #     # get predicted states for each model
        #     predicted_obs = model.get_prediction(obs, action, self.data_statistics)
        #     sum_rewards_per_seq = self.env.get_reward(predicted_obs, action)


        #   print('sum rew seq shape', sum_rewards_per_seq.shape)
        #   print('sum rew shape', sum_of_rewards.shape)

          
        # for action in candidate_action_sequences[0]:
        #   print('action here', action[None].shape)
        #   print('obs here', obs.shape)
        #   predicted_obs = model.get_prediction(obs, action[None], self.data_statistics)
        #   print('pred obs shape', predicted_obs.shape)


        # for action in candidate_action_sequences[0]:
        #   pred_sum = self.env.get_reward(predicted_obs, action[None])
        #   print('pred sum shape', pred_sum)
        #   # candidate_action_sequences[i, None, None])
        #   sum_of_rewards += pred_sum
        #   print('sum of rews shape here', sum_of_rewards.shape)

        # print('final sum of rews shape', sum_of_rewards)

            # mean_pred = self.evaluate_candidate_sequences