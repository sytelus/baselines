from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.deepq import defaults
from baselines.common.atari_wrappers import ClipRewardEnv, FrameStack


import gym
from podworld.envs import PodWorldEnv # import is needed to exec regiter in init
from baselines.common.atari_wrappers import TimeLimit, WarpFrame


def make_podworld(env_id:str, max_episode_steps=2000, clip_rewards=True, frame_stack=False):
    env = PodWorldEnv(obs_mode='R', agent_ray_count=84, agent_obs_height=84)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)        
    return env    


def main():
    exp_dir = './runs/podworld'

    # by default CSV logs will be created in OS temp directory
    logger.configure(dir=exp_dir, 
        format_strs=['stdout','log','csv','tensorboard'], log_suffix=None)

    # create Atari environment, use no-op reset, max pool last two frames
    env = make_podworld('podworld-v0')

    # by default monitor will log episod reward and log
    env = bench.Monitor(env, logger.get_dir())

    learn_params = defaults.atari()
    learn_params['checkpoint_path'] = exp_dir
    learn_params['checkpoint_freq'] = 100000 
    learn_params['print_freq'] = 10
    # learn_params['convs']=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    # learn_params['hiddens']=[256],    

    model = deepq.learn(
        env,
        total_timesteps=int(1e7),
        **learn_params
    )

    model.save('podworld_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
