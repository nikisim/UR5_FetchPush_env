import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import gym_UR5_FetchReach

import numpy as np

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = 'saved_models/UR5_FetchReach_test_action/FetchReach-v1/model_best.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make('gym_UR5_FetchReach/UR5_FetchReachEnv-v0', render=True)
    # env = gym.wrappers.RecordVideo(env, f"videos/FetchReach")#, episode_trigger = lambda x: x % 100 == 0)

    # get the env param
    observation, _ = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    success_episodes = 0
    for i in range(args.demo_length):
        observation, _ = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(30):
            # env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            print("Action,", action)
            # put actions into the environment
            observation_new, reward, truncated, terminanted, info = env.step(action)
            # print("Reward,", reward)
            done = truncated or terminanted
            obs = observation_new['observation']
            # if done:
            #     break
            
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
        if info['is_success'] == 1.0:
            success_episodes += 1
    
    print('******'*10)
    print(f"Success episodes: {success_episodes} out of {args.demo_length}, success rate = {success_episodes/args.demo_length}")
    print('******'*10)