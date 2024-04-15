from gym.envs.registration import register

register(
    id="gym_UR5_FetchPush/UR5_FetchPushEnv-v0",
    entry_point="gym_UR5_FetchPush.envs:UR5_FetchPushEnv",
)