from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ATSPGenerator(Generator):
    """Data generator for the Asymmetric Travelling Salesman Problem (ATSP)
    Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
    We satifsy the triangle inequality (TMAT class) in a batch

    Args:
        num_loc: number of locations (customers) in the TSP
        min_dist: minimum value for the distance between nodes
        max_dist: maximum value for the distance between nodes
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        dist_distribution: distribution for the distance between nodes
        tmat_class: whether to generate a class of distance matrix

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """

    def __init__(
        self,
        num_loc: int = 10,
        min_dist: float = 0.0,
        max_dist: float = 1.0,
        init_sol_type: str = "random",
        dist_distribution: int | float | str | type | Callable = Uniform,
        tmat_class: bool = True,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.tmat_class = tmat_class
        self.init_sol_type = init_sol_type

        # Distance distribution
        if kwargs.get("dist_sampler", None) is not None:
            self.dist_sampler = kwargs["dist_sampler"]
        else:
            self.dist_sampler = get_sampler("dist", dist_distribution, 0.0, 1.0, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
            self.dist_sampler.sample((batch_size + [self.num_loc, self.num_loc]))
            * (self.max_dist - self.min_dist)
            + self.min_dist
        )
        dms[..., torch.arange(self.num_loc), torch.arange(self.num_loc)] = 0
        log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            for i in range(self.num_loc):
                dms = torch.minimum(dms, dms[..., :, [i]] + dms[..., [i], :])
        return TensorDict({"cost_matrix": dms}, batch_size=batch_size)

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