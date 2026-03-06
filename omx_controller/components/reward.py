"""import math
class RewardFunction():
    def __init__(self,ball_pos, joint_pos, episode):
        super().__init__()
        self.joint_pos = joint_pos
        self.ball_pos = ball_pos
        self.episode = episode
        self.done = check_if_done(self.ball_pos, self.episode)
        self.reward = reward(self.ball_pos, self.joint_pos, self.done, self.episode)

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def distance_to_nearest_edge(ball_pos):
    return min(abs(ball_pos[0] - 0), abs(0.8 - ball_pos[0]), abs(ball_pos[1] - (-0.4)), abs(0.4 - ball_pos[1]))

def check_if_done(ball_pos, episode):
    
    if(ball_pos[2] < 0.45 or ball_pos[0] > 0.8 or ball_pos[0] < 0 or ball_pos[1] > 0.4 or ball_pos[1] < -0.4): # ball out of table
        return True
    elif episode == 2000: # in the table, timelimit = 2000
        return True
    return False

def reward(ball_pos, wrist_pos, done, timestep):
    # ---- constants ----
    w_hand = 1.0
    w_edge = 5.0
    w_success = 100.0
    w_time = 0.01

    d_hand_ball = euclidean_distance(ball_pos, wrist_pos)
    d_ball_edge = distance_to_nearest_edge(ball_pos)

    r = 0.0

    # Phase 1: approach ball
    r += -w_hand * d_hand_ball

    # Phase 2: push ball toward edge
    r += -w_edge * d_ball_edge
    # time penalty
    r += -w_time * timestep
    # Phase 3: success
    if done:
        if timestep <= 1999: # not force done
            r += w_success


    return r
"""
# import math
# import numpy as np

# class RewardFunction():
#     def __init__(self, ball_pos, joint_pos, timestep,
#                  target_origin=(0.2, -0.2),
#                  target_size=(0.3, 0.3),
#                  max_steps=2000):

#         self.ball_pos = ball_pos
#         self.joint_pos = joint_pos
#         self.timestep = timestep

#         self.target_origin = target_origin
#         self.target_size = target_size
#         self.max_steps = max_steps

#         self.done = check_if_done(
#             self.ball_pos,
#             self.timestep,
#             self.target_origin,
#             self.target_size,
#             self.max_steps
#         )

#         self.reward = reward(
#             self.ball_pos,
#             self.joint_pos,
#             self.done,
#             self.timestep,
#             self.target_origin,
#             self.target_size
#         )


# # ==============================
# # Utilities
# # ==============================

# def euclidean_distance(a, b):
#     return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# def is_inside_target(ball_pos, origin, size):
#     x_min, y_min = origin
#     width, height = size

#     x_max = x_min + width
#     y_max = y_min + height

#     return (
#         x_min <= ball_pos[0] <= x_max and
#         y_min <= ball_pos[1] <= y_max and
#         abs(ball_pos[2]) <= 0.02  # gần mặt đất
#     )


# def distance_to_target_center(ball_pos, origin, size):
#     x_min, y_min = origin
#     width, height = size

#     center = [
#         x_min + width / 2.0,
#         y_min + height / 2.0,
#         0.0
#     ]

#     return euclidean_distance(ball_pos, center)


# # ==============================
# # Done condition
# # ==============================

# def check_if_done(ball_pos, timestep, origin, size, max_steps):

#     # success
#     if is_inside_target(ball_pos, origin, size):
#         return True

#     # timeout
#     if timestep >= max_steps:
#         return True

#     return False


# # ==============================
# # Reward
# # ==============================

# def reward(ball_pos, wrist_pos, done, timestep, origin, size):

#     # ---- weights ----
#     w_hand = 5
#     w_target = 1
#     w_success = 200.0
#     w_time = 0.01

#     r = 0.0

#     # 1️ Encourage hand to reach ball
#     d_hand_ball = euclidean_distance(ball_pos, wrist_pos)
#     if np.isnan(d_hand_ball):
#         d_hand_ball = 1.0
#     r += -w_hand * d_hand_ball

#     # 2️ Encourage ball to move toward target center
#     d_target = distance_to_target_center(ball_pos, origin, size)
#     r += -w_target * d_target

#     # 3️ Time penalty
#     r += -w_time * timestep
#     if d_hand_ball < 0.03:
#         r += 5.0
#     # 4 Success reward
#     if done and is_inside_target(ball_pos, origin, size):
#         r += w_success

#     return r

import numpy as np
import math


class RewardFunction:
    def __init__(
        self,
        ball_pos,
        wrist_pos,
        prev_ball_pos,
        prev_wrist_pos,
        timestep,
        target_origin=(0.2, -0.2),
        target_size=(0.3, 0.3),
        max_steps=500,
    ):

        self.ball_pos = np.array(ball_pos)
        self.wrist_pos = np.array(wrist_pos)
        self.prev_ball_pos = np.array(prev_ball_pos)
        self.prev_wrist_pos = np.array(prev_wrist_pos)

        self.timestep = timestep
        self.target_origin = target_origin
        self.target_size = target_size
        self.max_steps = max_steps

        self.done = self.check_done()
        self.reward = self.compute_reward()

    # ==========================================================
    # Utilities
    # ==========================================================

    def target_center(self):
        x_min, y_min = self.target_origin
        w, h = self.target_size

        return np.array([
            x_min + w / 2.0,
            y_min + h / 2.0,
            0.0
        ])

    def is_success(self):
        x_min, y_min = self.target_origin
        w, h = self.target_size

        x_max = x_min + w
        y_max = y_min + h

        return (
            x_min <= self.ball_pos[0] <= x_max and
            y_min <= self.ball_pos[1] <= y_max and
            abs(self.ball_pos[2]) < 0.03
        )

    # ==========================================================
    # Done
    # ==========================================================

    def check_done(self):
        if self.is_success():
            return True

        if self.timestep >= self.max_steps:
            return True

        return False

    # ==========================================================
    # Reward
    # ==========================================================

    # def compute_reward(self):

    #     # Nếu chưa có prev state → không tính progress
    #     if self.prev_ball_pos is None or self.prev_wrist_pos is None:
    #         return 0.0

    #     d_hand = np.linalg.norm(self.ball_pos - self.wrist_pos)
    #     d_hand_prev = np.linalg.norm(self.prev_ball_pos - self.prev_wrist_pos)

    #     center = self.target_center()

    #     d_target = np.linalg.norm(self.ball_pos - center)
    #     d_target_prev = np.linalg.norm(self.prev_ball_pos - center)

    #     progress_hand = d_hand_prev - d_hand
    #     progress_target = d_target_prev - d_target

    #     w_hand = 10.0
    #     w_target = 20.0
    #     w_touch = 5.0
    #     w_success = 200.0

    #     r = 0.0

    #     r += w_hand * progress_hand
    #     r += w_target * progress_target

    #     r += -0.5 * d_hand
    #     r += -0.5 * d_target

    #     if d_hand < 0.03:
    #         r += w_touch

    #     if self.is_success():
    #         r += w_success

    #     return float(r)
    def compute_reward(self):

        if self.prev_wrist_pos is None:
            return 0.0

        r = 0.0

        center = self.target_center()

        d_hand = np.linalg.norm(self.ball_pos - self.wrist_pos)
        d_hand_prev = np.linalg.norm(self.ball_pos - self.prev_wrist_pos)

        d_target = np.linalg.norm(self.ball_pos - center)
        d_target_prev = np.linalg.norm(self.prev_ball_pos - center)

        progress_hand = d_hand_prev - d_hand
        progress_target = d_target_prev - d_target

        # reach ball
        r += 20 * progress_hand

        # encourage getting close
        r += -1.0 * d_hand

        # touching bonus
        if d_hand < 0.05:
            r += 5

            # only after touch → push ball
            r += 30 * progress_target

        # success
        if self.is_success():
            r += 200

        return r