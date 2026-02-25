"""
An example script showing how to train and test a RL policy for a differential robot to reach a goal using stable-baselines3
"""


from m0.learning.envs.goal_env import DiffDrivePointGoalEnv
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import time
import mujoco




if __name__ == "__main__":

    train = True

    # ---------- train ----------
    def make_env():
        return DiffDrivePointGoalEnv(
            xml_path="m0/assets/rl_scene.xml",
            frame_skip=5,
            max_duration=20,
            seed=0,
            render=True
        )


    if train:
        SEED = 2
        n_envs = 1
        vec_env = make_vec_env(make_env, n_envs=n_envs, seed=SEED)

        model = PPO(
            MlpPolicy,
            vec_env,
            verbose=1,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,          
            gae_lambda=0.95,
            ent_coef=0.0,
            device="cpu",
            tensorboard_log="m0/learning/tensorboard",
        )

        model.learn(total_timesteps=1_500_000, log_interval=5)
        model.save("m0/learning/model/model.zip")




    # ---------- test ----------
    else:
        # load model
        model = PPO.load("m0/learning/model/model.zip")
        # load env
        test_env = make_env()

        #### draw the goal position in mujoco viewer
        test_env.viewer.draw_point(np.array([test_env.goal[0], 
                                             test_env.goal[1], 
                                             0.03]))


        obs = test_env.reset()[0]
        test_env.render()

        ## run several episodes
        n_episodes = 10
        
        for ep in range(n_episodes):
            
            done = False
            ep_rew = 0.0

            while not done:
                
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = test_env.step(action)
                time.sleep(0.003)
                ep_rew += float(rew)
                done = bool(terminated or truncated)

                if done:
                    for i in range(200): # stop at the goal for seconds
                        test_env.data.ctrl[0] = 0.0
                        test_env.data.ctrl[1] = 0.0
                        mujoco.mj_step(test_env.model, test_env.data)
                        test_env.render()
                        time.sleep(0.001)

            print(f"Episode {ep+1}: Return={ep_rew}, Success={info.get('success', False)}")
            obs = test_env.reset()[0]
