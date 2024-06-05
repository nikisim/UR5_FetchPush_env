from gym.envs.registration import register

register(
    id="gym_UR5_FetchReach/UR5_FetchReachEnv-v0",
    entry_point="gym_UR5_FetchReach.envs:UR5_FetchReachEnv",
)