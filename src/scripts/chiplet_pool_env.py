"""
ChipletPoolEnv: OpenAI Gym environment wrapping Mozart chiplet pool optimization.

This module bridges the Mozart chiplet framework with the arch-gym optimization
infrastructure, enabling BO, GA, ACO, RL, PSO, DE algorithms for chiplet pool
composition search.

Usage:
    env = ChipletPoolEnv(virtual_nets=virtual_nets, n_chiplets=8,
                         objective="energy", database_file="llama_qwen_all_dram.csv")
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

import os
import sys
import copy
import time
import logging
import numpy as np

# Add arch-gym to path
ARCH_GYM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'oss-arch-gym'))
if ARCH_GYM_DIR not in sys.path:
    sys.path.insert(0, ARCH_GYM_DIR)

# Add chiplet_timeloop parent dir so local packages (timeloop/, etc.) are importable
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)

# Add timeloop_experiments for compute_area
_EXP_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

import gym
from gym import spaces

from chiplet_dataclass import ChipletConfig
from global_parameter import (
    arch_targets as DEFAULT_ARCH_TARGETS,
    glb_scales, pe_scales,
)
import global_parameter
from network_dataclass import VirtualNetwork
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
from cal_perf_phy_net import preload_database


# Encoding maps for discrete action space
ARCH_INDEX = {i: a for i, a in enumerate(DEFAULT_ARCH_TARGETS)}
GLB_INDEX = {i: g for i, g in enumerate(glb_scales)}
PE_INDEX = {i: p for i, p in enumerate(pe_scales)}
DRAM_INDEX = {i: d for i, d in enumerate(global_parameter.dram_options)}

N_ARCH = len(DEFAULT_ARCH_TARGETS)   # 3
N_GLB = len(glb_scales)              # 4
N_PE = len(pe_scales)                # 4

# Parameters per chiplet: arch, glb, pe_x, pe_y (DRAM decided by inner GA)
PARAMS_PER_CHIPLET = 4


class ChipletPoolEnv(gym.Env):
    """
    Gym environment for chiplet pool composition optimization.

    Action: MultiDiscrete array encoding N chiplets, each with 4 discrete params
            [arch_0, glb_0, pe_x_0, pe_y_0, arch_1, glb_1, pe_x_1, pe_y_1, ...]

    Observation: Array of per-workload metric values + aggregate metric.

    Reward: Negative of aggregate energy/EDP (minimization -> maximize negative).
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        virtual_nets,
        n_chiplets: int = 8,
        objective: str = "energy",
        database_file: str = "llama_qwen_all_dram.csv",
        cost_aware: bool = False,
        prev_best_genes=None,
        reward_formulation: str = "energy",
        n_workers: int = 8,
    ):
        super().__init__()

        self.virtual_nets = virtual_nets
        self.n_chiplets = n_chiplets
        self.objective = objective
        self.database_file = database_file
        self.cost_aware = cost_aware
        self.prev_best_genes = prev_best_genes
        self.reward_formulation = reward_formulation
        self.n_workers = n_workers

        # Action space: N chiplets x 4 params each (arch, glb, pe_x, pe_y)
        self.action_space = spaces.MultiDiscrete(
            [N_ARCH, N_GLB, N_PE, N_PE] * n_chiplets
        )

        # Observation space: per-workload metrics + aggregate
        n_obs = len(virtual_nets) + 1
        self.observation_space = spaces.Box(
            low=0, high=1e15, shape=(n_obs,), dtype=np.float64
        )

        # State tracking
        self.steps = 0
        self.best_value = float('inf')
        self.best_group = None
        self.last_obs = None

        # Logging
        self.logger = logging.getLogger('ChipletPoolEnv')

        # History for analysis
        self.eval_history = []

        # Pre-load database filtered to needed networks for fast evaluation
        needed_nets = set(vn.network_name for vn in virtual_nets)
        preload_database(database_file, needed_nets=needed_nets)

    def action_to_chiplet_group(self, action):
        """Convert flat action array to list of ChipletConfig objects."""
        chiplet_group = []
        for i in range(self.n_chiplets):
            base = i * PARAMS_PER_CHIPLET
            chiplet = ChipletConfig(
                arch_target=ARCH_INDEX[action[base]],
                global_buffer_size_scale=GLB_INDEX[action[base + 1]],
                pe_x_scale=PE_INDEX[action[base + 2]],
                pe_y_scale=PE_INDEX[action[base + 3]],
            )
            chiplet_group.append(chiplet)
        return chiplet_group

    def chiplet_group_to_action(self, chiplet_group):
        """Convert list of ChipletConfig objects to flat action array."""
        action = []
        arch_rev = {v: k for k, v in ARCH_INDEX.items()}
        glb_rev = {v: k for k, v in GLB_INDEX.items()}
        pe_rev = {v: k for k, v in PE_INDEX.items()}
        for chiplet in chiplet_group:
            action.extend([
                arch_rev[chiplet.arch_target],
                glb_rev[chiplet.global_buffer_size_scale],
                pe_rev[chiplet.pe_x_scale],
                pe_rev[chiplet.pe_y_scale],
            ])
        return np.array(action, dtype=np.int64)

    def _evaluate(self, chiplet_group):
        """Evaluate a chiplet group and return (aggregate_value, per_network_results)."""
        _, results = run_single_optimization(
            virtual_nets=self.virtual_nets,
            chiplet_group=chiplet_group,
            objective=self.objective,
            results_file=self.database_file,
            cost_aware=self.cost_aware,
            prev_best_genes=self.prev_best_genes,
            use_sequential=True,  # sequential GA to reuse in-process cache
            n_workers=self.n_workers,  # parallelize across networks via fork
        )
        avg_value, network_results = calculate_average_opt_value(results, self.objective)
        return avg_value, network_results, results

    def step(self, action):
        """Take one step: evaluate the chiplet pool defined by action."""
        self.steps += 1

        # Decode action to chiplet group
        chiplet_group = self.action_to_chiplet_group(action)

        # Evaluate
        start_time = time.perf_counter()
        avg_value, network_results, raw_results = self._evaluate(chiplet_group)
        eval_time = time.perf_counter() - start_time

        # Build observation
        per_net_values = [
            network_results[vn.get_unique_name()]['min_value']
            for vn in self.virtual_nets
        ]
        obs = np.array(per_net_values + [avg_value], dtype=np.float64)
        self.last_obs = obs

        # Compute reward (negative because we minimize energy/EDP)
        reward = -avg_value

        # Track best
        if avg_value < self.best_value:
            self.best_value = avg_value
            self.best_group = copy.deepcopy(chiplet_group)

        # Record history
        self.eval_history.append({
            'step': self.steps,
            'value': avg_value,
            'best_value': self.best_value,
            'eval_time': eval_time,
            'chiplet_ids': [c.get_identifier() for c in chiplet_group],
        })

        # Single-step episodes (each action is independently evaluated)
        done = True

        info = {
            'avg_value': avg_value,
            'network_results': network_results,
            'eval_time': eval_time,
            'chiplet_group': chiplet_group,
        }

        return obs, reward, done, info

    def reset(self):
        """Reset environment state."""
        self.steps = 0
        n_obs = len(self.virtual_nets) + 1
        self.last_obs = np.zeros(n_obs, dtype=np.float64)
        return self.last_obs

    def render(self, mode='human'):
        """Print current best solution."""
        if self.best_group is not None:
            print(f"Step {self.steps} | Best {self.objective}: {self.best_value:.4e}")
            for i, c in enumerate(self.best_group):
                print(f"  Chiplet {i}: {c.get_identifier()}")

    def get_search_space_size(self):
        """Return the total search space size."""
        per_chiplet = N_ARCH * N_GLB * N_PE * N_PE  # 3*4*4*4 = 192; this env's action space does not encode DRAM
        return per_chiplet ** self.n_chiplets

    def sample_random_action(self):
        """Sample a random valid action."""
        return self.action_space.sample()
