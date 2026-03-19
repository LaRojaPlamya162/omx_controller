# import copy
# import torch
# from omx_controller.models.IQL.network import QNetwork, VNetwork, Policy
# class IQLAgent:
#     def __init__(self, state_dim, action_dim,
#                  tau=0.7, beta=3.0, gamma=0.99, ema=0.005):

#         # Double Q
#         self.q1 = QNetwork(state_dim, action_dim)
#         self.q2 = QNetwork(state_dim, action_dim)

#         self.v = VNetwork(state_dim)
#         self.v_target = copy.deepcopy(self.v)  # target V

#         self.policy = Policy(state_dim, action_dim)

#         self.q_opt = torch.optim.Adam(
#             list(self.q1.parameters()) + list(self.q2.parameters()), 3e-4)
#         self.v_opt = torch.optim.Adam(self.v.parameters(), 3e-4)
#         self.pi_opt = torch.optim.Adam(self.policy.parameters(), 3e-4)

#         self.tau = tau
#         self.beta = beta
#         self.gamma = gamma
#         self.ema = ema
import copy
import torch
import torch.nn.functional as F
from omx_controller.models.IQL.network import QNetwork, ValueNetwork, Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class IQLAgent:
    def __init__(self, state_dim, action_dim):

        self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q_target = copy.deepcopy(self.q1)

        self.v = ValueNetwork(state_dim).to(DEVICE)
        self.actor = Actor(state_dim, action_dim).to(DEVICE)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.discount = 0.99
        self.tau = 0.005

        self.expectile = 0.7
        self.beta = 3.0

    def train(self, batch):
        s, a, r, s_next, done = batch

        s = s.to(DEVICE)
        a = a.to(DEVICE)
        r = r.to(DEVICE)
        s_next = s_next.to(DEVICE)
        done = done.to(DEVICE)

        # ---------------------
        # 1. Update V
        # ---------------------
        with torch.no_grad():
            q1 = self.q1(s, a)
            q2 = self.q2(s, a)
            q = torch.min(q1, q2)

        v = self.v(s)
        v_loss = expectile_loss(q - v, self.expectile).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # ---------------------
        # 2. Update Q
        # ---------------------
        with torch.no_grad():
            target_v = self.v(s_next)
            target_q = r + (1 - done) * self.discount * target_v

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------------------
        # 3. Update Actor
        # ---------------------
        with torch.no_grad():
            v = self.v(s)
            q = torch.min(self.q1(s, a), self.q2(s, a))
            advantage = q - v
            weight = torch.exp(advantage / self.beta).clamp(max=100.0)

        log_prob = self.actor.log_prob(s, a)
        actor_loss = -(weight * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------
        # 4. Soft update
        # ---------------------
        for param, target_param in zip(self.q1.parameters(), self.q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item()
        }
    
def expectile_loss(diff, tau):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * diff.pow(2)