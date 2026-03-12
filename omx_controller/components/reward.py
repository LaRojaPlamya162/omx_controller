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
        max_steps = 1000
        #max_steps=500,
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