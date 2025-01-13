from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        #empty ordered dict
        loss = OrderedDict()
    
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        loss_c = []
        loss_a = []

        no_steps = self.agent_params['num_critic_updates_per_agent_update']
        for i in range(no_steps):
          # ob_no, ac_na, next_ob_no, reward_n, terminal_n)
          crit = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
          # print('crit:', crit)
          loss_c.append(crit)

        # advantage = estimate_advantage(...)
        advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        n_steps = self.agent_params['num_actor_updates_per_agent_update']
        for i in range(n_steps):
          act = self.actor.update(ob_no, ac_na, advantages)
          print('act:', act)
          loss_a.append(act)

          loss['Loss_Critic'] = loss_c

          loss['Loss_Actor'] = loss_a


          train_log = {
              'Training Loss_Critic': loss['Loss_Critic'],
              'Training Loss_Actor': loss['Loss_Actor'],
          }

        print('loss value is:', loss)
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):

        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        v_s = self.critic.forward_np(ob_no)

        # 2) query the critic with next_ob_no, to get V(s')
        v_sp = self.critic.forward_np(next_ob_no)

        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        est_q = re_n + self.gamma * v_sp * (1 - terminal_n) # when done = true, cut off

        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        adv_n = est_q - v_s

        if self.standardize_advantages:
            adv_n = normalize(adv_n, adv_n.mean(), adv_n.std())
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
