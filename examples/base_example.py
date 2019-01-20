"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
import gym
from absl import flags
from pysc2.lib import point_flag
from pysc2.env import sc2_env

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS
FLAGS([__file__])


class BaseExample(object):
    def __init__(self, env_name, visualize=False, step_mul=None) -> None:
        super().__init__()
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul

    def run(self, num_episodes=1):
        env = gym.make(self.env_name)
        env.settings['visualize'] = self.visualize
        env.settings['step_mul'] = self.step_mul
        env.settings['agent_interface_format'] = sc2_env.parse_agent_interface_format(feature_screen=84,
                                                                                      feature_minimap=64,
                                                                                      rgb_screen=None,
                                                                                      rgb_minimap=None,
                                                                                      action_space="features",
                                                                                      use_feature_units=False)
        episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = env.reset()

            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, reward, done, _ = env.step(action)

            episode_rewards[ix] = env.episode_reward

        env.close()

        return episode_rewards

    def get_action(self, env, obs):
        raise NotImplementedError('Inherited classes must override get_action() method')
