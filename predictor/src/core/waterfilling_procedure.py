from __future__ import annotations

from typing import List

import numpy as np

from utilities.interpolate import LinearlyInterpolatedFunction


def waterfilling_procedure(inflow: float,
                           h: List[LinearlyInterpolatedFunction],
                           alpha: List[float],
                           beta: List[float]):
    deg = len(beta)
    sorted_beta = [(beta[i], i) for i in range(deg)]
    sorted_beta.sort()
    _, perm = zip(*sorted_beta)
    inv_perm = [0] * len(beta)
    for i in range(len(beta)):
        inv_perm[perm[i]] = i

    # Find maximum r s.t. sum_{i=0,...,r-1} max { z | h_i(z) <= beta[r-1] } <= inflow
    max_z_r = []
    max_z_rp1 = None
    r = 0
    while r < len(beta):
        # compute sum_{i=0,...,r}  max { z | h_i(z) <= beta[r] }
        beta_rp1 = beta[perm[r]]
        max_z_rp1 = [
            h[perm[i]].max_t_below_bound(beta_rp1, default=0.)
            for i in range(r + 1)
        ]
        if sum(max_z_rp1) > inflow:
            break
        r += 1
        max_z_r = max_z_rp1

    if r < len(beta) and sum(max_z_rp1[i] for i in range(r)) <= inflow:
        z = np.zeros(len(beta))
        for i in range(r):
            z[inv_perm[i]] = max_z_rp1[i]
        z[inv_perm[r]] = inflow - sum(z[inv_perm[i]] for i in range(r))
        assert abs(sum(z) - inflow) < .005
        return z
    else:
        z = np.zeros(len(beta))
        assert len(max_z_r) == r
        b_prime = inflow - sum(max_z_r)
        assert r > 1
        for i in range(r):
            z[inv_perm[i]] = max_z_r[i] + alpha[inv_perm[i]] / sum(alpha[inv_perm[j]] for j in range(r)) * b_prime
        assert abs(sum(z) - inflow) < .005
        return z
