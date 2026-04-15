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
        target_center_pos=(0.2, -0.2),
        target_size=(0.2, 0.2),
        max_steps = 500
        #max_steps=500,
    ):
        self.ball_pos = np.array(ball_pos)
        self.wrist_pos = np.array(wrist_pos)
        self.prev_ball_pos = np.array(prev_ball_pos)
        self.prev_wrist_pos = np.array(prev_wrist_pos)
        self.timestep = timestep
        self.target_center_pos = target_center_pos
        #self.target_origin = target_origin
        self.target_size = target_size
        self.max_steps = max_steps
        #self.done = self.check_done()
        self.reward = self.compute_reward()

    def target_center(self):
        return np.array([
            self.target_center_pos[0],
            self.target_center_pos[1],
            0.0
        ])

    # ==========================================================
    # Done
    # ==========================================================

    def is_success(self):
        cx, cy = self.target_center_pos
        w, h = self.target_size

        x_min = cx - w / 2.0
        x_max = cx + w / 2.0
        y_min = cy - h / 2.0
        y_max = cy + h / 2.0

        return (
            x_min <= self.ball_pos[0] <= x_max and
            y_min <= self.ball_pos[1] <= y_max and
            abs(self.ball_pos[2]) < 0.15
        )

    def check_done(self):
        if self.is_success():
            return True

        if self.check_out_of_time():
            return True
        if not self.is_ball_in_playground():
            return True
        return False
    def check_out_of_time(self):
        return self.timestep >= self.max_steps
    def is_ball_in_playground(self, 
                         playground_size=(0.8, 0.8),   # ← thay đổi ở đây
                         playground_center=(0.0, 0.0)):
        """playground_size = (width_x, depth_y)"""
        half_w = playground_size[0] / 2.0
        half_d = playground_size[1] / 2.0
        
        x_min = playground_center[0] - half_w
        x_max = playground_center[0] + half_w
        y_min = playground_center[1] - half_d
        y_max = playground_center[1] + half_d
        
        in_bounds = (x_min <= self.ball_pos[0] <= x_max and 
                    y_min <= self.ball_pos[1] <= y_max)
        
        not_too_high = abs(self.ball_pos[2]) < 0.15   # cho phép bóng bay lên một chút
        
        return in_bounds and not_too_high
    def compute_reward(self):
        if self.prev_ball_pos is None or self.prev_wrist_pos is None:
            return 0.0

        r = 0.0
        center = self.target_center()

        dist_hand = np.linalg.norm(self.ball_pos - self.wrist_pos)
        dist_hand_prev = np.linalg.norm(self.prev_ball_pos - self.prev_wrist_pos)

        dist_target = np.linalg.norm(self.ball_pos[:2] - center[:2])
        dist_target_prev = np.linalg.norm(self.prev_ball_pos[:2] - center[:2])

        # ====================== Phase-based Shaping ======================
        if dist_hand > 0.08:
            # ==================== REACH PHASE ====================
            progress_hand = dist_hand_prev - dist_hand
            r += 10.0 * progress_hand
            r += -0.5 * dist_hand                    # giảm từ -1.2 → -0.5

        else:
            # ==================== PUSH PHASE ====================
            progress_target = dist_target_prev - dist_target
            
            # Chỉ thưởng progress_target khi đang giữ bóng (tránh fake progress)
            r += 18.0 * progress_target
            r += -0.8 * dist_target


        # ====================== Touching Bonus ======================
        TOUCH_THRESHOLD = 0.08

        # 1. One-time bonus khi VỪA chạm bóng
        if dist_hand < TOUCH_THRESHOLD and dist_hand_prev >= TOUCH_THRESHOLD:
            r += 8.0

        # 2. Small dense bonus khi đang giữ bóng (giảm để tránh farm)
        elif dist_hand < TOUCH_THRESHOLD:
            r += 0.8                              # giảm từ 1.2 → 0.8

        # ====================== Penalty & Success ======================
        if not self.is_ball_in_playground():
            r -= 20.0

        if self.is_success():
            r += 120.0

        # Time penalty nhẹ
        r -= 0.03

        # ====================== Clip reward (RẤT QUAN TRỌNG cho IQL/SAC) ======================
        r = np.clip(r, -20.0, 20.0)

        return r