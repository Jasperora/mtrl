# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Class to interface with an Experiment"""

from typing import List
import imageio
import numpy as np
from mtenv.utils.types import ObsType

from mtrl.agent import utils as agent_utils
from mtrl.env.vec_env import VecEnv  # type: ignore[attr-defined]
from mtrl.experiment import multitask
from mtrl.utils.types import ConfigType, TensorType
import wandb


class Experiment(multitask.Experiment):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        """Experiment Class to manage the lifecycle of a multi-task model.

        Args:
            config (ConfigType):
            experiment_id (str, optional): Defaults to "0".
        """
        super().__init__(config, experiment_id)

    def get_action_when_evaluating_vec_env_of_tasks(
        self, multitask_obs: ObsType, modes: List[str]
    ) -> TensorType:
        agent = self.agent
        # modes: ['base' or 'interpolation' or 'extrapolation']
        with agent_utils.eval_mode(agent):
            # agent: mtrl.agent.distral.Agent
            # action = agent.sample_action(multitask_obs=multitask_obs, modes=modes)

            # original version
            action = agent.select_action(multitask_obs=multitask_obs, modes=modes)

        return action

    def evaluate_vec_env_of_tasks(self, vec_env: VecEnv, step: int, episode: int):
        """Evaluate the agent's performance on the different environments,
        vectorized as a single instance of vectorized environment.

        Since we are evaluating on multiple tasks, we track additional metadata
        to track which metric corresponds to which task.

        Args:
            vec_env (VecEnv): vectorized environment.
            step (int): step for tracking the training of the agent.
            episode (int): episode for tracking the training of the agent.
        """
        for mode in self.eval_modes_to_env_ids:
            self.logger.log(f"{mode}/episode", episode, step)

        # episode_reward, finish, done = [
        #     np.full(shape=vec_env.num_envs, fill_value=fill_value)
        #     for fill_value in [0.0, False, False]
        # ]  # (num_envs, 1)

        episode_reward, mask, done = [
            np.full(shape=vec_env.num_envs, fill_value=fill_value)
            for fill_value in [0.0, 1.0, False]
        ]

        multitask_obs = vec_env.reset()  # (num_envs, 9, 84, 84)
        offset = self.config.experiment.num_eval_episodes

        # for rendering
        imgs = []
        # while not np.all(done):
        for _ in range(1000):
            action = self.get_action_when_evaluating_vec_env_of_tasks(
                multitask_obs=multitask_obs, modes=vec_env.mode
            )
            multitask_obs, reward, done, _ = vec_env.step(action)
            mask = mask * (1 - done.astype(int))
            # print("mask: ", mask)
            # print(episode_reward)
            episode_reward += reward * mask

            # render img
            img = vec_env.render()
            imgs.append(img)
            # print(done)
        # while not np.all(finish):
        # for _ in range(1000):
        #     action = self.get_action_when_evaluating_vec_env_of_tasks(
        #         multitask_obs=multitask_obs, modes=vec_env.mode
        #     )
        #     multitask_obs, reward, done, _ = vec_env.step(action)
        #     img = vec_env.render()
        #     episode_reward += reward * (1-finish)
        #     imgs.append(img)
        #     finish |= done

        start_index = 0
        for mode in self.eval_modes_to_env_ids:
            if self.eval_modes_to_env_ids[mode] != None:
                num_envs = len(self.eval_modes_to_env_ids[mode])
                self.logger.log(
                    f"{mode}/episode_reward",
                    episode_reward[start_index : start_index + offset * num_envs].mean(),
                    step,
                )
                # print("self.eval_modes_to_env_ids[mode]: ", self.eval_modes_to_env_ids[mode])
                # print("mode: ", mode)
                for _current_env_index, _current_env_id in enumerate(
                    self.eval_modes_to_env_ids[mode]
                ):
                    self.logger.log(
                        f"{mode}/episode_reward_env_index_{_current_env_index}",
                        episode_reward[
                            start_index
                            + _current_env_index * offset : start_index
                            + (_current_env_index + 1) * offset
                        ].mean(),
                        step,
                    )
                    self.logger.log(
                        f"{mode}/env_index_{_current_env_index}", _current_env_id, step
                    )
                    rew = episode_reward[
                            start_index
                            + _current_env_index * offset : start_index
                            + (_current_env_index + 1) * offset
                        ].mean()
                    wandb.log({"eval_episode_reward_{}".format(_current_env_id): rew}, step=step)
                start_index += offset * num_envs
        self.logger.dump(step)
        
        imgs = np.array(imgs) # [episode_len, env_num, width, height, channel]
        imgs = imgs.transpose(1,0,2,3,4)
        imgs = np.concatenate(imgs, axis=1)
        imgs = imgs.transpose(0,3,1,2) 
        wandb.log(
            {"video": wandb.Video(imgs[-100:], fps=10, caption="result.gif")}, 
            step=step
        )
        # for i in range(len(imgs)):
        #     # imageio.mimsave(
        #     #     uri="result_{:02d}_{}.gif".format(i,step),
        #     #     ims=imgs[i],
        #     #     format="GIF",
        #     # )

        #     wandb.log({"video": wandb.Video(imgs[i], fps=10, caption="result_{:02d}.gif".format(i))}, step=step)

