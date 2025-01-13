from collections import OrderedDict
import numpy as np
import time

import gym
from numpy.random import logseries
import torch

from rob831.infrastructure import pytorch_util as ptu
from rob831.infrastructure.logger import Logger
from rob831.infrastructure import utils
import pickle 

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
# MAX_VIDEO_LEN = 40  # we overwrite this in the code below


def make_env(env_name):
    if env_name == 'Ant-v2':
        return gym.make(env_name)
    else:
        return gym.make(env_name)


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = make_env(self.params['env_name'])
        self.env.reset(seed=seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        # self.MAX_VIDEO_LEN = self.params['ep_len']
        self.MAX_VIDEO_LEN = 40

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.metadata['render_fps']

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        print('agent', agent_class)
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self.params['batch_size']
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # HW1: implement this function below

            # log/save
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy,
            batch_size,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ``` 
        path = '/content/gdrive/MyDrive/hw_16831/16831-F24-HW/hw1/rob831/expert_data'
        
        print('666 bug', itr)
        #for first iteration:
        if itr == 0:
          loaded_paths = pickle.load(open(self.params['expert_data'], 'rb'))
          return loaded_paths, 0, None
           # return paths, 0 (steps dis batch), none yet
        

        # else use collect pollicy:
        
        # (2) collect `self.params['batch_size']` transitions
        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
      
        # if utils.get_pathlength(paths) == self.params['ep_len']:
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, self.params['batch_size'], self.params['ep_len'])

        for i in range(len(paths)):
            if utils.get_pathlength(paths[i]) != self.params['ep_len']:
                print('wtf', utils.get_pathlength(paths[i]), self.params['ep_len'])
            # assert utils.get_pathlength(paths[i]) == self.params['ep_len']
      

        # utils.get_pathlength(paths) == self.params['ep_len']
        # print('path_length', utils.get_pathlength(paths))

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        print('bug running too fast', self.log_video)
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, self.MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths


    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            s_o, s_a, s_r, s_no, s_t = self.agent.sample(self.params['train_batch_size'])
         
            #convert
            t_so = ptu.from_numpy(s_o)
            t_sa = ptu.from_numpy(s_a)
            t_sr = ptu.from_numpy(s_r)
            t_sno = ptu.from_numpy(s_no)
            t_st = ptu.from_numpy(s_t)
            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
          
            # train agent (using sampled data from replay buffer)
            train_log = self.agent.train(t_so, t_sa, t_sr, t_sno, t_st)  # HW1: implement this function below

            all_logs.append(train_log)
        return all_logs
        

    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        print("is list?", paths)
        for i in range(len(paths)):
            # obs = self.agent.get_action(paths[i]["observation"])

            # paths([i]["actions"]) = 

            expert_action = expert_policy.get_action(paths[i]['observation'])
            paths[i]['action'] = expert_action
         # relabel the collected obs with actions from a provided expert policy


        # for i in range(self.training_returns):
        #   mod = MLPPolicy(self.current_policy, paths[i]["observation"])
        #   mod.get_action()

        #   paths = MLPPolicy.
        #   training_returns = get_action(self.expert_policy, paths[i]["action"])
        # utils.Path(obs, image_obs, acs, rewards, next_obs, terminal)
            # if relabel_with_expert and itr>=start_relabel_with_expert:
            #     paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below 

        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        print('shut up bug', self.log_video, train_video_paths)
        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, self.MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()

    ####################################
    ###### DRAFT
    ####################################
        #  train_log = self.collect_training_trajectories(
        #         self.itr,
        #         self.initial_expertdata,
        #         self.expert_policy,
        #         data 
        #     )
        #     all_logs.append(train_log)
        # logs =  self.train_agent()
        #         data = 
    #         ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch =  self.collect_training_trajectories(
    #             itr,
    #             initial_expertdata,
    #             collect_policy,
    #             self.params['batch_size']
    #         )  
    # ):

                    #     param itr:
                # :param load_initial_expertdata:  path to expert data pkl file
                # :param collect_policy:  the current policy using which we collect data
                # :param batch_size:  the number of transitions we collect

        # """
        # if no_iterations == 0:
        #   use collect_policy to collect data
        # else:
        #   load data --> return loaded_paths, 0, none
  
        # """

        """
        if no_iterations == 0:
          load data --> return loaded_paths, 0, none
        else:
          use collect_policy to collect data
  
        """

         # paths, envsteps_this_batch = self.collect_training_trajectories(
          #   self.itr,
          #   self.initial_expertdata,
          #   self.collect_policy
          #   self.params['train_batch_size']
          # )
          # utils.sample_trajectories(self.env, self.collect_policy, self.params['train_batch_size'], self.params['ep_len'])