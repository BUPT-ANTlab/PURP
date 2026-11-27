"""Node and STOP action scores over the corrected GNN3 state encoder."""
import torch
from torch import nn
from metro.models.policy import MetroPolicy


class StopPolicy(MetroPolicy):
    def __init__(self, cfg, agent, shared_net):
        super().__init__(cfg, agent, shared_net)
        self.stop_head = nn.Sequential(nn.Linear(shared_net.output_policy_road_size, 32),
                                       nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        state, _, mask, stage = self.shared_net(x)
        return self.distribution(state, mask), stage

    def distribution(self, state, mask):
        node_logits = self.policy_road_head(state)
        stop_logit = self.stop_head(state.mean(dim=1))
        logits = torch.cat([node_logits, stop_logit], dim=1)
        if logits.shape != mask.shape or not torch.all(mask.bool().any(dim=1)):
            raise ValueError("Invalid action mask")
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask.bool(), -torch.inf))

    def select_action(self, x, mean_action=False):
        distribution, _ = self(x)
        return distribution.logits.argmax(dim=1) if mean_action else distribution.sample()

    def get_log_prob_entropy(self, x, action):
        distribution, _ = self(x)
        action = action.long().reshape(-1)
        return distribution.log_prob(action).unsqueeze(1), distribution.entropy().unsqueeze(1)
