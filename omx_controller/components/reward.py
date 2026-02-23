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
import math

class RewardFunction():
    def __init__(self, ball_pos, joint_pos, timestep,
                 target_origin=(0.2, -0.2),
                 target_size=(0.3, 0.3),
                 max_steps=2000):

        self.ball_pos = ball_pos
        self.joint_pos = joint_pos
        self.timestep = timestep

        self.target_origin = target_origin
        self.target_size = target_size
        self.max_steps = max_steps

        self.done = check_if_done(
            self.ball_pos,
            self.timestep,
            self.target_origin,
            self.target_size,
            self.max_steps
        )

        self.reward = reward(
            self.ball_pos,
            self.joint_pos,
            self.done,
            self.timestep,
            self.target_origin,
            self.target_size
        )


# ==============================
# Utilities
# ==============================

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def is_inside_target(ball_pos, origin, size):
    x_min, y_min = origin
    width, height = size

    x_max = x_min + width
    y_max = y_min + height

    return (
        x_min <= ball_pos[0] <= x_max and
        y_min <= ball_pos[1] <= y_max and
        abs(ball_pos[2]) <= 0.02  # gần mặt đất
    )


def distance_to_target_center(ball_pos, origin, size):
    x_min, y_min = origin
    width, height = size

    center = [
        x_min + width / 2.0,
        y_min + height / 2.0,
        0.0
    ]

    return euclidean_distance(ball_pos, center)


# ==============================
# Done condition
# ==============================

def check_if_done(ball_pos, timestep, origin, size, max_steps):

    # success
    if is_inside_target(ball_pos, origin, size):
        return True

    # timeout
    if timestep >= max_steps:
        return True

    return False


# ==============================
# Reward
# ==============================

def reward(ball_pos, wrist_pos, done, timestep, origin, size):

    # ---- weights ----
    w_hand = 1.0
    w_target = 5.0
    w_success = 200.0
    w_time = 0.01

    r = 0.0

    # 1️ Encourage hand to reach ball
    d_hand_ball = euclidean_distance(ball_pos, wrist_pos)
    r += -w_hand * d_hand_ball

    # 2️ Encourage ball to move toward target center
    d_target = distance_to_target_center(ball_pos, origin, size)
    r += -w_target * d_target

    # 3️ Time penalty
    r += -w_time * timestep

    # 4 Success reward
    if done and is_inside_target(ball_pos, origin, size):
        r += w_success

    return r