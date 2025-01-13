import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize
# from torch._C import R

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # storage = np.zeros((4, observations.shape[0]))
        # advs = np.zeros(observations)
        # advs = np.zeros(len(rewards_list))
        # print('reward list shape', rewards_list.shape)
        store = []
   
        # TODO: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy


        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log

        # print('obs shape', observations.shape)
        # print('actions', observations.shape)
        # print('whats in rewards', len(rewards_list))
        # print('whats in rewards down', len(rewards_list[0]))
        q_vals = self.calculate_q_vals(rewards_list) # list with 43 rows 
        # store.append(q_vals)


        # print('stored q vals', len(store))
        # print('stored q vals first dim', len(store[0]))
        advs = self.estimate_advantage(observations, rewards_list, q_vals, terminals)
   
        train_log  = self.actor.update(observations, actions, advs, q_vals)
        # raise NotImplementedError

        return train_log

       # print(f"Shape of adv: {np.shape(adv)}")  # Check the shape
          # print(f"Type of adv: {type(adv)}")  # Check the type
          # advs[i] = np.array(adv)



    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # l_rews = len(rewards_list) # no rows
        # count = 0 
        # for i in rewards_list:
        #     count += len(i)
        #     # print('sanity count', count)


        r = len(rewards_list)
        q_values =[]

        # l = rewards_list[0]
        # c = len(rewards_list[0])
        # print('rows', r)
        # print('cols', c)
        # storage = []
        # print('yo que whats ur size?', len(q_values))

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.
        if not self.reward_to_go:
            #use the whole traj for each timestep
          for i in range(len(rewards_list)):
              q_values.extend(self._discounted_return(rewards_list[i]))
              # print('q_vals dr', len(q_values))


        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
          # for i in reversed(range(r)):
          for i in range(len(rewards_list)):
            q_values.extend(self._discounted_cumsum(rewards_list[i]))

        # print('rewards list length', r)
        # print('rewards list cols', len(rewards_list[0]))

        # print('q_vals list elements', len(q_values))

        return np.array(q_values)  # return an array

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """
        l_rews = len(rewards_list)
        storage = []

        print('type', type(obs))
        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            # print('terminals what r u', terminals)

            values_normalized = self.actor.run_baseline_prediction(obs)
            values = unnormalize(values_normalized, q_values.mean(), q_values.std())
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_normalized.ndim == q_values.ndim
            # assert values.shape == q_values.shape
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values

            # raise NotImplementedError
            # values = self.actor.run_baseline_prediction(obs) #take maean
            
            # values = values_normalized

            # if values_normalized has same mean and std as q_values.mean() and q_values.std
          
            # if values.mean() != q_values.mean() & values.std() != q_values.std():
            #   values_
            # assert values_normalized.std() == q_values.std()
               
            print('values size', values.size)
            # values_normalized
            
            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rewards = np.concatenate(rewards_list)
                print('what are rewars', rewards.shape)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.

                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                  # print('terminals are', terminals[i])
                  if (terminals[i] == True) | (terminals[i] == False):
                    # if (terminals[i] == True):
                    #   # this is the last state in the trajectory 
                        if (i + 1) <= batch_size:
                          d_t = rewards[i] + (self.gamma * values[i + 1])- values[i]
                          advantages[i] = d_t + (self.gamma*self.gae_lambda*advantages[i+1])
                        else:
                            d_t = rewards[i] - values[i]
                            advantages[i] = d_t

                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula

                    # raise NotImplementedError

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                # raise NotImplementedError
                advantages = q_values - values
                # advantages = values - q_values
                # v -  self.gae_lambda*q_values 

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()
            # print('advantages size def', advantages.shape)

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            norm_adv = normalize(advantages, advantages.mean(), advantages.std())
            # raise NotImplementedError
            advantages = norm_adv
            print('standardized advanteges', self.standardize_advantages)


        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards): # discounted reward from full trajectory
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        #l_rews = T
        l_rews = len(rewards)

        #create empty array size of list
        discounted_returns = np.zeros(len(rewards))
        # np.zeros(l_rews)

        # TODO: create discounted_returns
        # raise NotImplementedError
        for t in reversed(range(0, l_rews)): #start at l_rews, stop at 0
          d_r =  0
          for j in range(l_rews):
            d_r += (self.gamma)**(j)*rewards[j] #discounted_returns[t+1]
            # d_r += rewards[t] + (self.gamma)**(j - t)*rewards[j] #discounted_returns[t+1]
            discounted_returns[t] = d_r 
            # discounted_returns[t] = np.sum_(t=0)^l_rews, self.gamma^t, discounted_returns[0]
          # print('d_r', discounted_returns)
          # print('dr size ye', discounted_returns)

        return discounted_returns

    def _discounted_cumsum(self, rewards): # discounted reward from "reward-to-go" trajectory
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        
        #l_rews = T
        l_rews = len(rewards)
       

        #create empty array size of list
        discounted_cumsums = np.zeros(len(rewards))


     # TODO: create `discounted_cumsums`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        for t in reversed(range(l_rews)): #start at l_rews, stop at 0
          d_c =  0
          for j in range(t, l_rews):
            d_c += ((self.gamma)**(j - t))*rewards[j] 
            discounted_cumsums[t] = d_c
   
        # for t in reversed(range(l_rews)):
        #     cumsum = rewards[t] + (self.gamma)*cumsum
        #     discounted_cumsums[t] = cumsum
            # discounted_cumsums[t] = rewards[t] + (self.gamma)*discounted_cumsums[t+1]

        # print('lrews should be', l_rews)
        # print('thus, dc should be', len(discounted_cumsums))
        return discounted_cumsums


        #### -------- DRAFT ------------------------------------------------------------------

                    # dc.append(discounted_cumsums[t])
            # print('dc full', discounted_cumsums[t])
            # .sum_(t_n=t)^l_rews, self.gamma^(t_n - t)*rewards[t_n]

            # discounted_cumsums[t] = np.sum_(t=t_n)^l_rews, self.gamma^(t_n - t), discounted_cumsums[t_n]
            # print('d_c', discounted_cumsums)

        # raise NotImplementedError
                  # if terminals[i] == 1:# last in trajectory
                          #   advantages[i] = self.gae_lambda*advantages[i]
                          #   break
                          # else: # not last state thus, i + 1
                          #   advantages[i] = self.gae_lambda*advantages[i+1]

            #  rewards_list[i] + (q_values[i+1])
            # raise NotImplementedError
            # for i in range(r):
            #   for j in range(c):
            #     q_values[i][j] = self._discounted_return(rewards_list[i][j])
            #     print('is this an array', q_values[i][j])

                        # rewards_list[i] + (q_values[i+1])

            #  for i in range(r):
            #   for j in range(c):
            #     q_values[i][j] = self._discounted_cumsum(rewards_list[i][j])
            #     print('is that an array', q_values[i][j])

          # q_values = self._discounted_cumsum(rewards_list)

            # storage.append(q_values)
            # print('q_vals dc', len(storage[0]))
              # storage.append(q_values)
              # print('q_val r2g', q_values)
              # print('yo dc que', q_values[0])
            # raise NotImplementedError
        
                      # storage.append(q_values)
              # print('q_val no_r2g', q_values)

              # print('yo dr que', q_values[0])
      # l = len(rewards_list[i]) # length of list at row i
        # q_values = storage
          
       # Convert advantages list to a NumPy array 
        # advs = np.zeros((len(advantages),len(max(advantages, key = lambda x: len(x)))))  # Create an empty array with dimensions (number of episodes, length of longest episode)
        # for i, adv in enumerate(advantages):
        #   advs[i,:len(adv)] = adv  # Pad with 0's 

        # print('advantages here', advs)

        # adv = np.array(adv)
        # print('stor', storage)
        # advs = np.ndarray(storage)
        # rl_size = rewards_list.shape
        # print('rl shape', rl_size)
                # obs, rews, next_obs, terminals = self.replay_buffer.sample_random_data(observations.shape[0])
        # print('new rews', rews)
        # for i in range(len(q_vals)):
          # storage = q_vals[i]
        # print('what q_vals size', len(q_vals))
        # q_vals = np.ndarray(q_vals)
        # print('what q_vals size', len(q_vals[0]))

        # for i in range(len(q_vals)):
        # print('what  storage', storage)
        # for i in range(len(q_vals)):
        # q_vals_arr = np.ndarray(q_vals)
        # print('is array?', len(q_vals))
        
        # for i in range(len(q_vals)):
        # for i in range(len(rewards_list)):
                  # if t == (l_rews-1):
          #   discounted_cumsums[t] = rewards[t]
          #   # print('dc empty', discounted_cumsums[t])
          #   # dc.append(discounted_cumsums[t])
          # else:
          # d_c = 0
          # for j in range(t, l_rews):
          #   d_c += (self.gamma)**(j-t)*rewards[j]
          #   discounted_cumsums[t] = d_c
                  # print('discount', discounted_cumsums.size)
        #  np.zeros(l_rews)

        #init
        # discounted_cumsums[-1] = rewards[-1] 

                    # l = len(rewards_list[i]) # length of list at row i
            # for j in reversed(range(l-1)):
                  # for i in reversed(range(r)):
            #  q_values = self._discounted_return(rewards_list[i])
          # for i in reversed(range(r)):
                        # l = len(rewards_list[i]) # length of list at row i
              # for j in reversed(range(l)):
