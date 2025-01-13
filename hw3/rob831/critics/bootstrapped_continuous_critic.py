from .base_critic import BaseCritic
from torch import nn
from torch import optim

from rob831.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        # print(f"Input shape to critic network: {obs.shape}")  # Add this line to print the shape
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # dim = reward_n.shape
        # print('reward shape:', dim)
        episodes = (self.num_grad_steps_per_target_update * self.num_target_updates)
        count = 0
        steps = self.num_grad_steps_per_target_update
        # loss = 0
        # v_sp = 0
        loss_val = 0

        # loss_val = []
        #make tensors
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)


        # shape = next_ob_no.shape
        # # print('obs first dim', len(shape))
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward

        for e in range(episodes):
          v_s = self.forward(ob_no) # first state in episode

          if count == 0 | count % steps == 0:
            # if len(shape) == 1:
            v_sp = self.forward(next_ob_no)
            # else:
            #   v_sp = self.forward_np(ptu.to_numpy(next_ob_no))

            # print('output critic shape', v_sp.shape)
            target = reward_n + self.gamma * v_sp * (1 - terminal_n)
            loss_val = self.loss(v_s, target)
            # loss_val.append(loss_comp) 

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

          # if terminal_n[step]:
          count +=1
          #   break

          return loss_val.item()


########################## DRAFT ####################


        # # print('obs shape', ob_no.shape)
        # # print('next obs shape', next_ob_no.shape)
        # # print('rew shape', reward_n.shape)
        # # print('term shape', terminal_n.shape)

        # shape = next_ob_no.shape
        # # print('obs first dim', len(shape))


        # # print('obs shape', ob_no.shape)
        # # print('next obs shape', next_ob_no.shape)
        # # print('rew shape', reward_n.shape)
        # # print('term shape', terminal_n.shape)

          # if count == 0: # for first step
          #   self.optimizer.zero_grad()
          #   v_sp = self.critic.forward(next_ob_no)
          #   target = reward_n + self.gamma * v_sp * (1 - terminal_n) # desired
          #   loss = self.loss(v_sp, target).squeeze()

          # else:

                  
        # for i in range(grad_steps):
        # for e in range(grad_steps):
          # for i in range(len(reward_n)):
          # v_sp = self.forward_np(ob_no).squeeze()
            
          # target = reward_n + self.gamma * v_sp * (1 - terminal_n) # desired
    
          # loss_p = self.loss(v_sp, target)
          # loss.append(loss_p)  

          # if count == 0 | count % steps == 0:
          #   # for i in range(len(reward_n)):
          #     self.optimizer.zero_grad()
          #     # TODO: Implement the pseudocode below: do the following (
          #     # self.num_grad_steps_per_target_update * self.num_target_updates)
          #     # times:
          #     # every self.num_grad_steps_per_target_update steps (which includes the
          #     # first step), recompute the target values by
          #     #     a) calculating V(s') by querying the critic with next_ob_no
          #     # next_ob_no = (next_ob_no.reshape[0], -1) 

          #     #convert to tensor:
          #     # ten_n_obs = ptu.from_numpy(next_ob_no)
          #     v_sp = self.forward(next_ob_no)#remove first dim
              
          #     if v_sp[1] > 1:
          #       v_sp = v_sp.squeeze
                
              
          #     # 
          #     # else:
          #     #   v_sp = self.forward(ten_n_obs)

              
          #     # tensor_shape = self.forward(ten_n_obs).shape
          #     # reward_shape = reward_n.shape
          #     # print('reward shape', reward_shape)
          #     # print('tensor shape', tensor_shape)
          #     # if r > 1 and c > 1

          #     # if len(tensor_shape) > 1 and tensor_shape[1]:
          #     # if tensor_shape[1] > 1: #if the first dimension (cols) is > 4
          #     #   v_sp = self.forward(ten_n_obs).squeeze(4) #remove first dim
          #     # else:
          #     #   v_sp = self.forward(ten_n_obs).squeeze()

          #     # v_sp = v_sp.squeeze(1)
              
          #     #     b) and computing the target values as r(s, a) + gamma * V(s')
          #     # every time, update this critic using the observations and targets
          #             # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
          #     #       to 0) when a terminal state is reached
          #     # HINT: make sure to squeeze the output of the critic_network to ensure
          #     #       that its dimensions match the reward
          #     target = reward_n + self.gamma * v_sp * (1 - terminal_n) # desired

          #     loss_p = self.loss(v_sp, target)
          #     loss.append(loss_p)    

          # loss.backward()
          # self.optimizer.step()
          # count+=1  # update count
