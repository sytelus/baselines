from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari, FrameStack
from baselines.deepq import defaults
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule


# - agent sees 3rd and 4th frames
# - greyscale
# - reward clipping/huber loss
# - learning rate needs to be lower for more complex games
# - tf.variance_scaling_initializer with scale=2

def main():
    exp_dir = './runs/breakout'

    # by default CSV logs will be created in OS temp directory
    logger.configure(dir=exp_dir, 
        format_strs=['stdout','log','csv','tensorboard'], log_suffix=None)

    # create Atari environment, use no-op reset, max pool last two frames
    env = make_atari('BreakoutNoFrameskip-v4')

    # by default monitor will log episod reward and log
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    learn_params = defaults.atari()
    learn_params['checkpoint_path'] = exp_dir
    learn_params['checkpoint_freq'] = 100000 
    learn_params['print_freq'] = 10
    learn_params['exploration_scheduler'] = PiecewiseSchedule([ \
                (0,        1.0),
                (int(1e6), 0.1),
                (int(1e7), 0.01)
            ], outside_value=0.01)

    model = deepq.learn(
        env,

        # below are defaults
        #convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        #hiddens=[256],

        total_timesteps=int(3e7),
        **learn_params
    )

    model.save('breakout_model.pkl')
    env.close()

if __name__ == '__main__':
    main()
