"""
ChipletPoolEstimator: scikit-learn BaseEstimator wrapper for Bayesian Optimization
of chiplet pool composition using arch-gym / scikit-optimize.

Usage with BayesSearchCV:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer

    search_space = ChipletPoolEstimator.build_search_space(n_chiplets=8)
    estimator = ChipletPoolEstimator(virtual_nets=virtual_nets, n_chiplets=8, ...)
    opt = BayesSearchCV(estimator, search_space, n_iter=100, scoring='neg_mean_squared_error')
    opt.fit(X_dummy)

Direct usage with skopt.Optimizer:
    optimizer = build_skopt_optimizer(n_chiplets=8)
    for i in range(n_iter):
        suggestion = optimizer.ask()
        value = estimator.evaluate_point(suggestion)
        optimizer.tell(suggestion, value)
"""

import os
import sys
import copy
import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# Add chiplet_timeloop parent dir so local packages (timeloop/, etc.) are importable
_CHIPLET_TL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _CHIPLET_TL_DIR not in sys.path:
    sys.path.insert(0, _CHIPLET_TL_DIR)
_EXP_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'timeloop_experiments'))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from sklearn.base import BaseEstimator

from chiplet_dataclass import ChipletConfig
from global_parameter import (
    arch_targets as DEFAULT_ARCH_TARGETS,
    glb_scales, pe_scales,
)
from network_dataclass import VirtualNetwork
from utility_functions import calculate_average_opt_value
from chiplet_sel import run_single_optimization
from cal_perf_phy_net import preload_database


class ChipletPoolEstimator(BaseEstimator):
    """
    sklearn-compatible estimator that evaluates chiplet pool configurations.
    Designed for use with scikit-optimize's BayesSearchCV or Optimizer.
    """

    def __init__(
        self,
        virtual_nets=None,
        n_chiplets: int = 8,
        objective: str = "energy",
        database_file: str = "llama_qwen_all_dram.csv",
        cost_aware: bool = False,
        prev_best_genes=None,
        # Per-chiplet parameters (used by BayesSearchCV set_params)
        **chiplet_params,
    ):
        self.virtual_nets = virtual_nets
        self.n_chiplets = n_chiplets
        self.objective = objective
        self.database_file = database_file
        self.cost_aware = cost_aware
        self.prev_best_genes = prev_best_genes

        # Store chiplet parameters
        self.chiplet_params = chiplet_params

        # Tracking
        self.eval_count = 0
        self.eval_history = []
        self.best_value = float('inf')
        self.best_group = None

        self.logger = logging.getLogger('ChipletPoolEstimator')

    def _decode_params_to_group(self) -> List[ChipletConfig]:
        """Decode stored chiplet_params dict into a list of ChipletConfig."""
        chiplets = []
        for i in range(self.n_chiplets):
            arch = self.chiplet_params.get(f'arch_{i}', DEFAULT_ARCH_TARGETS[0])
            glb = self.chiplet_params.get(f'glb_{i}', glb_scales[0])
            pe_x = self.chiplet_params.get(f'pe_x_{i}', pe_scales[0])
            pe_y = self.chiplet_params.get(f'pe_y_{i}', pe_scales[0])
            chiplets.append(ChipletConfig(
                arch_target=arch,
                global_buffer_size_scale=int(glb),
                pe_x_scale=int(pe_x),
                pe_y_scale=int(pe_y),
            ))
        return chiplets

    def evaluate_group(self, chiplet_group: List[ChipletConfig]) -> float:
        """Evaluate a chiplet group directly. Returns aggregate metric value."""
        _, results = run_single_optimization(
            virtual_nets=self.virtual_nets,
            chiplet_group=chiplet_group,
            objective=self.objective,
            results_file=self.database_file,
            cost_aware=self.cost_aware,
            prev_best_genes=self.prev_best_genes,
            use_sequential=True,  # sequential GA to reuse in-process cache
            n_workers=8,  # parallelize across networks via fork
        )
        avg_value, _ = calculate_average_opt_value(results, self.objective)
        return avg_value

    def fit(self, X=None, y=None):
        """Evaluate the current chiplet parameters (called by BayesSearchCV)."""
        self.eval_count += 1
        start_time = time.perf_counter()

        chiplet_group = self._decode_params_to_group()
        value = self.evaluate_group(chiplet_group)

        eval_time = time.perf_counter() - start_time

        # Track best
        if value < self.best_value:
            self.best_value = value
            self.best_group = copy.deepcopy(chiplet_group)

        self.eval_history.append({
            'eval': self.eval_count,
            'value': value,
            'best_value': self.best_value,
            'eval_time': eval_time,
            'chiplet_ids': [c.get_identifier() for c in chiplet_group],
        })

        return self

    def score(self, X=None, y=None):
        """Return negative value (sklearn maximizes score, we minimize energy)."""
        chiplet_group = self._decode_params_to_group()
        value = self.evaluate_group(chiplet_group)
        return -value

    def get_params(self, deep=False):
        """Return all parameters (required by sklearn)."""
        params = {
            'virtual_nets': self.virtual_nets,
            'n_chiplets': self.n_chiplets,
            'objective': self.objective,
            'database_file': self.database_file,
            'cost_aware': self.cost_aware,
            'prev_best_genes': self.prev_best_genes,
        }
        params.update(self.chiplet_params)
        return params

    def set_params(self, **params):
        """Set parameters (called by BayesSearchCV during optimization)."""
        for key in ['virtual_nets', 'n_chiplets', 'objective', 'database_file',
                     'cost_aware', 'prev_best_genes']:
            if key in params:
                setattr(self, key, params.pop(key))
        # Remaining params are chiplet parameters
        self.chiplet_params.update(params)
        return self

    @staticmethod
    def build_search_space(n_chiplets: int):
        """Build scikit-optimize search space for BayesSearchCV."""
        from skopt.space import Categorical

        space = {}
        for i in range(n_chiplets):
            space[f'arch_{i}'] = Categorical(DEFAULT_ARCH_TARGETS)
            space[f'glb_{i}'] = Categorical(glb_scales)
            space[f'pe_x_{i}'] = Categorical(pe_scales)
            space[f'pe_y_{i}'] = Categorical(pe_scales)
        return space


def build_skopt_optimizer(n_chiplets: int, n_initial_points: int = 10, acq_func: str = "EI"):
    """
    Build a skopt.Optimizer for direct ask/tell loop optimization.

    Returns:
        optimizer: skopt.Optimizer instance
        dimensions: list of dimensions (for reference)
    """
    from skopt import Optimizer
    from skopt.space import Categorical

    dimensions = []
    dim_names = []
    for i in range(n_chiplets):
        dimensions.append(Categorical(DEFAULT_ARCH_TARGETS, name=f'arch_{i}'))
        dimensions.append(Categorical(glb_scales, name=f'glb_{i}'))
        dimensions.append(Categorical(pe_scales, name=f'pe_x_{i}'))
        dimensions.append(Categorical(pe_scales, name=f'pe_y_{i}'))
        dim_names.extend([f'arch_{i}', f'glb_{i}', f'pe_x_{i}', f'pe_y_{i}'])

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        random_state=42,
    )

    return optimizer, dim_names


def decode_skopt_point(point: list, n_chiplets: int) -> List[ChipletConfig]:
    """Convert a skopt suggestion (flat list) to list of ChipletConfig.
    4 params per chiplet: arch, glb, pe_x, pe_y.
    """
    chiplets = []
    for i in range(n_chiplets):
        base = i * 4
        chiplets.append(ChipletConfig(
            arch_target=point[base],
            global_buffer_size_scale=int(point[base + 1]),
            pe_x_scale=int(point[base + 2]),
            pe_y_scale=int(point[base + 3]),
        ))
    return chiplets
