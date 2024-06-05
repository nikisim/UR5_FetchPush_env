from gym.envs.registration import register

register(
    id="gym_UR5_FetchReach_as_real/UR5_FetchReach_as_realEnv-v0",
    entry_point="gym_UR5_FetchReach_as_real.envs:UR5_FetchReach_as_realEnv",
)