import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        q_value = self.critic.qa_values(observation)
        action = np.argmax(q_value, axis=1)

        return action.squeeze()


####### DRAFT ####################

        # obs_t = (observation)
                # print('Argmax policy q_value Type:', q_value.dtype)

        # print('Argmax policy Action Type:', action.dtype)
