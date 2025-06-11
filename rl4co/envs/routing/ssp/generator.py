from typing import Callable, Union

from tensordict.tensordict import TensorDict
import torch
import numpy as np
import hashlib
import random
import math
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SSPGenerator(Generator):

    def __init__(
        self,
        num_loc: int = 10,
        fixed_len: int = 8,
        min_loc: float = 0,
        max_loc: float = 1,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.fixed_len = fixed_len

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
        fixed_len = self.fixed_len
        codes = generate_batch_superstring_data(batch_size, self.num_loc, fixed_len)        
        return TensorDict({"codes": codes}, batch_size=batch_size)

def generate_batch_superstring_data(batch_size, num_str, str_dim, alphabet_size=2):
    # Generate random strings
    batch_data = torch.randint(0, alphabet_size, (batch_size, num_str, str_dim))
    
    # Generate random overlap masks
    overlap_mask = torch.rand(batch_size, num_str - 1) > 0.5
    overlap_lengths = torch.randint(1, str_dim // 2 + 1, (batch_size, num_str - 1))
    
    # Generate index tensors for efficient slicing
    overlap_indices = torch.arange(str_dim).expand(batch_size, num_str - 1, str_dim)
    overlap_mask_expanded = overlap_mask.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)
    overlap_lengths_expanded = overlap_lengths.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)

    # Generate a mask for the overlap regions
    overlap_region_mask = (overlap_indices < overlap_lengths_expanded) & overlap_mask_expanded
    
    # Copy the values to the overlap region
    previous_strings = batch_data[:, :-1, :].clone()
    for i in range(str_dim):
        current_mask = overlap_region_mask[:, :, i]
        selected_overlap_index_at_i = (str_dim - overlap_lengths + i).view(-1,1) % str_dim
        selected_overlap = previous_strings.view(-1, str_dim).gather(1, selected_overlap_index_at_i).view(batch_size, num_str - 1)
        batch_data[:, 1:, i][current_mask] = selected_overlap[current_mask]
    
    # Shuffle the num_str dimension
    perm = torch.rand(batch_size, num_str).argsort(dim=1)
    batch_data = batch_data[torch.arange(batch_size).unsqueeze(1), perm]
    
    return batch_data.float()



class SSPkoptGenerator(Generator):

    def __init__(
        self,
        num_loc: int = 10,
        fixed_len: int = 8,
        init_sol_type: str = "random",
        **kwargs,
    ):
        self.num_loc = num_loc
        self.fixed_len = fixed_len
        self.init_sol_type = init_sol_type

    def generate_batch_superstring_data(batch_size, num_str, str_dim, alphabet_size=2):
        # Generate random strings
        batch_data = torch.randint(0, alphabet_size, (batch_size, num_str, str_dim))
        
        # Generate random overlap masks
        overlap_mask = torch.rand(batch_size, num_str - 1) > 0.5
        overlap_lengths = torch.randint(1, str_dim // 2 + 1, (batch_size, num_str - 1))
        
        # Generate index tensors for efficient slicing
        overlap_indices = torch.arange(str_dim).expand(batch_size, num_str - 1, str_dim)
        overlap_mask_expanded = overlap_mask.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)
        overlap_lengths_expanded = overlap_lengths.unsqueeze(-1).expand(batch_size, num_str - 1, str_dim)

        # Generate a mask for the overlap regions
        overlap_region_mask = (overlap_indices < overlap_lengths_expanded) & overlap_mask_expanded
        
        # Copy the values to the overlap region
        previous_strings = batch_data[:, :-1, :].clone()
        for i in range(str_dim):
            current_mask = overlap_region_mask[:, :, i]
            selected_overlap_index_at_i = (str_dim - overlap_lengths + i).view(-1,1) % str_dim
            selected_overlap = previous_strings.view(-1, str_dim).gather(1, selected_overlap_index_at_i).view(batch_size, num_str - 1)
            batch_data[:, 1:, i][current_mask] = selected_overlap[current_mask]
        
        # Shuffle the num_str dimension
        perm = torch.rand(batch_size, num_str).argsort(dim=1)
        batch_data = batch_data[torch.arange(batch_size).unsqueeze(1), perm]
        
        return batch_data.float()

    def _generate(self, batch_size) -> TensorDict:
        batch_size = batch_size[0] if isinstance(batch_size, list) else batch_size
        fixed_len = self.fixed_len
        codes = generate_batch_superstring_data(batch_size, self.num_loc, fixed_len)  
        # Generate distance matrices inspired based on overlap regions between strings
        
         # Initialize cost matrix
        cost_matrix = torch.full((batch_size, self.num_loc, self.num_loc), fixed_len, dtype=torch.float32)

        # Compute overlap: cost[i][j] = fixed_len - max overlap from code[i] -> code[j]
        for shift in range(1, fixed_len):  # start from 1 (minimum overlap length)
            suffix = codes[:, :, -shift:]  # [B, N, shift]
            prefix = codes[:, :, :shift]   # [B, N, shift]

            # Expand and compare every (i,j) pair
            suffix_exp = suffix.unsqueeze(2).expand(-1, -1, self.num_loc, -1)  # [B, N, N, shift]
            prefix_exp = prefix.unsqueeze(1).expand(-1, self.num_loc, -1, -1)  # [B, N, N, shift]

            match = (suffix_exp == prefix_exp).all(dim=-1)  # [B, N, N]
            overlap_length = shift

            # Update cost only if overlap increases
            cost_matrix = torch.where(
                match, 
                fixed_len - overlap_length,  # smaller cost for larger overlap
                cost_matrix
            )

        # Set diagonal to large number to avoid self-loops
        inf = 1e6
        diag_idx = torch.arange(self.num_loc, device=codes.device)
        cost_matrix[:, diag_idx, diag_idx] = inf

        return TensorDict({"cost_matrix": cost_matrix, "codes": codes}, batch_size=batch_size)

    def _get_initial_solutions(self, cost_matrix):
        batch_size = cost_matrix.size(0)

        if self.init_sol_type == "random":
            set = torch.rand(batch_size, self.num_loc).argsort().long()
            rec = torch.zeros(batch_size, self.num_loc).long()
            index = torch.zeros(batch_size, 1).long()

            for i in range(self.num_loc - 1):
                rec.scatter_(1, set.gather(1, index + i), set.gather(1, index + i + 1))

            rec.scatter_(1, set[:, -1].view(-1, 1), set.gather(1, index))

        elif self.init_sol_type == "greedy":
            device = cost_matrix.device
            candidates = torch.ones(batch_size, self.num_loc, device=device).bool()
            rec = torch.zeros(batch_size, self.num_loc, device=device).long()
            selected_node = torch.zeros(batch_size, 1, device=device).long()
            candidates.scatter_(1, selected_node, 0)

            for i in range(self.num_loc - 1):
                # Use distance matrix instead of coordinates
                dists = cost_matrix.gather(
                    1, selected_node.unsqueeze(-1).expand(batch_size, 1, self.num_loc)
                ).squeeze(1)
                dists[~candidates] = 1e5

                next_selected_node = dists.min(-1)[1].view(-1, 1)
                rec.scatter_(1, selected_node, next_selected_node)
                candidates.scatter_(1, next_selected_node, 0)
                selected_node = next_selected_node

        else:
            raise NotImplementedError()

        return rec.expand(batch_size, self.num_loc).clone()