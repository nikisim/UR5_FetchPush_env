import torch
from rl_modules.models import actor
from arguments import get_args
import gymnasium as gym
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
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)#, render_mode="human")
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

    # for dataset
    observations_ = []
    actions_ = []
    rewards_ = []
    next_observations_ = []
    terminals_ = []
    truncations_ = []

    dataset_length = 15_000
    for i in range(dataset_length):
        observation, _ = env.reset()
        # start to do the demo
        # obs = observation['observation']
        g = observation['desired_goal']
        for t in range(50):
            # env.render()
            inputs = process_inputs(observation['observation'], g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, terminated, truncated,info = env.step(action)
            done = terminated or truncated
            #add to dataset
            observations_.append(np.concatenate((observation['observation'],
                                                 observation['desired_goal'],
                                                 observation['achieved_goal'])).astype(np.float32))
            next_observations_.append(np.concatenate((observation_new['observation'],
                                                      observation_new['desired_goal'],
                                                      observation_new['achieved_goal'])).astype(np.float32))
            actions_.append(action.astype(np.float32))
            rewards_.append(reward.astype(np.float32))
            terminals_.append(done)
            # truncations_.append(trunc)
            observation = observation_new
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))

    dataset = {
            'observations': np.array(observations_),
            'actions': np.array(actions_),
            'rewards': np.array(rewards_),
            'next_observations': np.array(next_observations_),
            'terminals': np.array(terminals_),
            # 'truncations': np.array(truncations_)
        }
    np.save('datasets/FetchPickAndPlaceDense.npy', dataset)