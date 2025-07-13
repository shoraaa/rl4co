from functools import lru_cache, cached_property
from typing import Optional, Tuple

import torch

from tensordict import TensorDict
from torch import Tensor
from tqdm import trange

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive.decoder import (
    NonAutoregressiveDecoder,
)
from rl4co.utils.decoding import Sampling
from rl4co.utils.ops import batchify, get_distance_matrix, unbatchify
from rl4co.models.zoo.deepaco.antsystem import AntSystem

class FocusedACO(AntSystem):
    """
    Focused Ant Colony Optimization (FACO) algorithm.
    This class extends the AntSystem class to implement the FACO algorithm.
    """
    
    def __init__(
        self,
        log_heuristic: Tensor,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.95,
        Q: Optional[float] = None,
        pheromone: Optional[Tensor | int] = None,
        use_local_search: bool = False,
        use_nls: bool = False,
        n_perturbations: int = 1,
        local_search_params: dict = {},
        perturbation_params: dict = {},
    ):
        self.batch_size = log_heuristic.shape[0]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.Q = 1 / self.n_ants / self.decay if Q is None else Q

        self.log_heuristic = log_heuristic

        if pheromone is None or isinstance(pheromone, int):
            self.pheromone = torch.ones_like(log_heuristic)
            self.pheromone.fill_(pheromone if isinstance(pheromone, int) else 1)
        else:
            assert pheromone.shape == log_heuristic.shape
            self.pheromone = pheromone

        self.final_actions = self.final_reward = None
        self.final_reward_cache: dict = {}

        self.use_local_search = use_local_search
        assert not (use_nls and not use_local_search), "use_nls requires use_local_search"
        self.use_nls = use_nls
        self.n_perturbations = n_perturbations
        self.local_search_params = local_search_params.copy()  # needs to be copied to avoid side effects
        self.perturbation_params = perturbation_params.copy()

        self._batchindex = torch.arange(self.batch_size, device=log_heuristic.device)

    
    def _update_pheromone(self, actions, reward):
    # Initialize Δpheromone
        delta_pheromone = torch.zeros_like(self.pheromone)

        from_node = actions[:, :, :-1]  # shape: [B, A, L-1]
        to_node = actions[:, :, 1:]     # shape: [B, A, L-1]

        mapped_reward = self._reward_map(reward).detach()  # [B, A]

        batch_action_indices = self._batch_action_indices(
            self.batch_size, actions.shape[-1] - 1, reward.device
        )  # shape: [B, L-1]

        # Get best ant per batch
        best_ant_idx = mapped_reward.argmax(dim=1)  # [B]

        for b in range(self.batch_size):
            best = best_ant_idx[b].item()
            fr = from_node[b, best]  # [L-1]
            to = to_node[b, best]    # [L-1]
            delta_pheromone[b, fr, to] += mapped_reward[b, best]

        # Decay + update
        self.pheromone *= self.decay
        self.pheromone += delta_pheromone

        # Per-batch tau_min and tau_max
        # rho = 1.0 - self.decay
        # p_best = 0.1
        # cand_count = self.pheromone.shape[-1]
        # avg = cand_count / 2.0
        # p = pow(p_best, 1.0 / cand_count)

        # min_vals, _ = reward.min(dim=1)         # shape: (B,)
        # cost = min_vals.abs()                   # |min reward[b]|

        # # vectorized τ_max and τ_min
        # tau_max = 1.0 / (cost * rho)            # shape: (B,)
        # tau_min = torch.minimum(
        #     tau_max,
        #     tau_max * (1 - p) / ((avg - 1) * p)
        # )                                        # shape: (B,)

        # # clamp pheromone per‐batch in one go
        # # expand to match pheromone’s [B, N, N] shape
        # tau_min = tau_min.view(-1, 1, 1)
        # tau_max = tau_max.view(-1, 1, 1)

        # self.pheromone = torch.clamp(
        #     self.pheromone,
        #     min=tau_min,
        #     max=tau_max
        # )
