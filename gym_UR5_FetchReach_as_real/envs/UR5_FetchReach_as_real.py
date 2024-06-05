import time
import math
import random
import os
import numpy as np
import pybullet as p
import pybullet_data

from utilities import Camera
from tqdm import tqdm
from robot import UR5Robotiq85
import gym
from gym import error, spaces
from gym.utils import seeding

def goal_distance(goal_a, goal_b):
    assert len(goal_a) == len(goal_b)
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class UR5_FetchReach_as_realEnv(gym.Env):

    SIMULATION_STEP_DELAY = 1 / 250.

    def __init__(self, render=False) -> None:
        super(UR5_FetchReach_as_realEnv, self).__init__()

        camera = Camera((1, 1, 1),
                        (0, 0, 0),
                        (0, 0, 1),
                        0.1, 5, (320, 320), 40)
        camera = None
        robot = UR5Robotiq85((0, 0, 0), (0, 0, 0))
        self.robot = robot
        self.vis = render
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        p.setTimeStep(1/250.)

        table_size = (0.5, 0.5)  # Length and width of the table
        table_height = 0.03 + 0.05 / 2

        target_position = self.random_position_on_table(0.5,0.5, table_height)

        # Create a visual shape (red sphere) for the target
        target_visual_shape = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
        self.target_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_visual_shape, basePosition=target_position)


        # For calculating the reward
        self.distance_threshold = 0.05

        self._max_episode_steps = 300
        self._elapsed_steps = None

        self.seed()
        obs = self._get_obs()
        self.action_space = spaces.Box(-0.15, 0.15, shape=(3,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    def get_max_steps(self):
        return self._max_episode_steps

    # Function to generate a random position on the table
    def random_position_on_table(self, table_length, table_width, table_height):
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(0.2, 0.5) # # Height of the table surface
        return [x, y, z]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    # def read_debug_parameter(self):
    #     # read the value of task parameter
    #     x = p.readUserDebugParameter(self.xin)
    #     y = p.readUserDebugParameter(self.yin)
    #     z = p.readUserDebugParameter(self.zin)
    #     roll = p.readUserDebugParameter(self.rollId)
    #     pitch = p.readUserDebugParameter(self.pitchId)
    #     yaw = p.readUserDebugParameter(self.yawId)
    #     gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

    #     return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action):
        """
        new_action: (dx,dy) -diff for x and y axis for end-effector
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """ 

        # print("Action:",action)
        new_action = np.clip(action, self.action_space.low, self.action_space.high)
        # new_action *= 0.2
        rpy = np.array([0,math.pi/2,math.pi/2,0])
        action = np.concatenate((new_action,rpy))
        
        self.robot.move_ee(action[:-1], 'end')
        self.robot.move_gripper(action[-1])
        
        # Step the simulation 20 times to maintain the control frequency of 25 Hz
        for _ in range(60):   # 20 simulation steps with a time step of 0.002 seconds
            self.step_simulation()
        
        truncation = False

        obs = self._get_obs()
        info = {}
        info['is_success'] = self._is_success(obs['achieved_goal'], self.goal)
        # if self._check_done(obs) == False:
        if info['is_success']:
            termination = True
        else:
            termination = False
        # else:
        #     termination = self._check_done(obs)

        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        # print("Reward", reward)
        # print("Info", info)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncation = True

        return obs, reward, termination, truncation, info

    # Function to calculate Euclidean distance
    def euclidean_distance(self, position1, position2):
        return np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        new_target_position_target = self.random_position_on_table(0.5,0.5, 0.03 + 0.05 / 2 + 0.15)
        p.resetBasePositionAndOrientation(self.target_body, new_target_position_target, [0, 0, 0, 1])
        return new_target_position_target
    
    def _sample_puck_pos(self):
        """Samples a new puck pos and returns it.
        """
        new_target_position_puck = self.random_position_on_table(0.5,0.5, 0.03 + 0.05 / 2) 
        p.resetBasePositionAndOrientation(self.puckId, new_target_position_puck, [0, 0, 0, 1])
        return new_target_position_puck

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        return -d

    # def get_observation(self):
    #     obs = dict()
    #     if isinstance(self.camera, Camera):
    #         rgb, depth, seg = self.camera.shot()
    #         obs.update(dict(rgb=rgb, depth=depth, seg=seg))
    #     else:
    #         assert self.camera is None
    #     obs.update(self.robot.get_joint_obs())
    #     # print(obs)
    #     return obs

    def get_ee_target_diff(self, ee_pos, target_pos):
        """Get difference betwee end-effector and target position over all axes [x,y,z]
        """
        dx = target_pos[0] - ee_pos[0]
        dy = target_pos[1] - ee_pos[1]
        dz = target_pos[2] - ee_pos[2]
       
        return np.array([dx,dy,dz])
    
    def _get_obs(self):
        robot_pos = self.robot.get_joint_obs()
        
        # puck_position, puck_orientation = p.getBasePositionAndOrientation(self.puckId)
        des_goal, _ = p.getBasePositionAndOrientation(self.target_body)

        # Get the position and orientation of the puck
        # puck_euler_orientation = p.getEulerFromQuaternion(puck_orientation)

        # Get the linear and angular velocity of the puck
        # puck_linear_velocity, puck_angular_velocity = p.getBaseVelocity(self.puckId)

        # diff_array = self.get_ee_puck_diff(robot_pos['ee_pos'], des_goal)

        obs = np.concatenate((
            np.array(robot_pos['ee_pos'], dtype='float32'),
            np.array(robot_pos['ee_vel'], dtype='float32'),
            # np.array(robot_pos['positions'], dtype='float32'),
            # np.array(robot_pos['velocities'], dtype='float32'),
            # np.array(diff_array, dtype='float32'),
            # current puck pos
            # np.array(puck_orientation, dtype='float32'),
            # puck orientation in Euler angles
            # np.array(puck_euler_orientation, dtype='float32'),
            # puck linear velocity
            # np.array(puck_linear_velocity, dtype='float32'),
            # puck angular velocity
            # np.array(puck_angular_velocity, dtype='float32'),
            )
        )
        achieved_goal = np.array(robot_pos['ee_pos']).copy()
        return {
            'observation': obs.copy(),
            'achieved_goal': np.array(achieved_goal).copy(),
            'desired_goal': np.array(des_goal).copy(),
        }

    # check if the puck outside the table
    def _check_done(self, obs):
        if obs['achieved_goal'][0] < -0.3 or obs['achieved_goal'][0] > 0.3:
            done = True
        elif obs['achieved_goal'][1] < -0.3 or obs['achieved_goal'][1] > 0.3:
            done = True
        else:
            done = False
        return done

    def reset(self):
        self._elapsed_steps = 0
        self.robot.reset()
        self.is_success = False

        self.goal = self._sample_goal().copy()
        # self.puck_pos =  self._sample_puck_pos().copy()
        obs = self._get_obs()

        while np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) <= 2 * self.distance_threshold:
            self.goal = self._sample_goal().copy()
            # self.puck_pos =  self._sample_puck_pos().copy()
            obs = self._get_obs()
        return obs, {}

    def close(self):
        p.disconnect(self.physicsClient)
