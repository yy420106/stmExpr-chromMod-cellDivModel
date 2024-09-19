import math
from typing import Any, Optional

import numba as nb
import numpy as np
import sympy as sp

__all__ = [
    "calc_Pme23",
    "calc_E",
    "calc_alpha",
    "calc_omega",
    "calc_theta",
    "calc_time_to_next_repl_after_ev",
    "calc_time_to_next_repl_after_ccc",
    "calc_prot_fixed_points",
]
# JIT compile: calc_Pme23, calc_E, calc_alpha, calc_omega, calc_theta
# these functions will be used in JIT-compiled Gillespie Stochastic Sampling Algorithm (SSA)


@nb.njit(signature_or_function=nb.float64(nb.int32[:]), cache=True)
def calc_Pme23(meState: np.ndarray[Any, int]) -> float:
    """
    Compute the percentage of repressive methylation marks (me2/me3) over target gene locus.

    Parameters
    ----------
    meState : NDArray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.

    Returns
    -------
    Pme23 : flotat64
       Ratio of repressive methylation modification.
    """

    Pme23 = np.sum(meState >= 2) / meState.size

    return Pme23


@nb.njit(signature_or_function=nb.float64[:](nb.int32[:], nb.float64, nb.float64), cache=True)
def calc_E(meState: np.ndarray[Any, int], rho: float, e_distal: float) -> np.ndarray[Any, float]:
    """
    Compute enhancement gain by neighbors and distal interaction for methylation propensities.

    Parameters
    ----------
    meState : NDArray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.
    rho : float64
        Model free parameter. Activation capacity of PRC2 by 2-methylation relative to 3-methylation.
    e_distal : float64
        Model free parameter. Distal gain.

    Returns
    -------
    E : NDArray[float64], shape (H,)
        Methylation enhancement gain by adjacent neighbors of each histone.
    """

    # initialize output
    E = np.empty(shape=(meState.size,), dtype=np.float64)

    for idx in range(meState.size):
        if idx % 2 == 0:
            neighbor_meState = np.concatenate(
                (meState[max(idx - 2, 0) : idx], meState[idx + 1 : min(idx + 4, meState.size)])
            )
        else:
            neighbor_meState = np.concatenate(
                (meState[max(idx - 3, 0) : idx], meState[idx + 1 : min(idx + 3, meState.size)])
            )
        E[idx] = rho * np.sum(neighbor_meState == 2) + np.sum(neighbor_meState == 3) + e_distal

    return E


@nb.njit(signature_or_function=nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), cache=True)
def calc_alpha(cell_cycle: float, cc0: float, mu: float, alpha_lim: float) -> float:
    """
    Compute the stable cycle-dependent gene activation level.

    Parameters
    ----------
    cell_cycle : float64
        Model free parameter. Cell cycle.
    cc0 : float64
        Model free parameter. Normalized cell cycle.
    mu : float64
        Model free parameter. Exponential factor.
    alpha_lim : float64
        Model free parameter. Lower bound of cycle-coupled STM gene activation level.

    Returns
    -------
    alpha : float64
        Cycle-dependent gene activation level.
    """

    alpha = max(math.pow(cc0 / cell_cycle, mu), alpha_lim)

    return alpha


@nb.njit(signature_or_function=nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), cache=True)
def calc_omega(cell_cycle: float, A: float, B: float, omega_lim: float) -> float:
    """
    Compute the cycle-dependent PRC2 activation level, used in indirect interaction test.

    Parameters
    ----------
    cell_cycle : float64
        Model free parameter. Cell cycle.
    A : float64
        Model free parameter. Absolute of slope on exponential.
    B : float64
        Model free parameter. Intercept on exponential.
    omega_lim : float64
        Model free parameter. Upper bound of cycle-dependent PRC2 activation level.

    Returns
    -------
    omega : float64
        Cycle-dependent PRC2 activation level.
    """

    omega = omega_lim / (1 + math.exp(-A * cell_cycle + B))

    return omega


@nb.njit(signature_or_function=nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), cache=True)
def calc_theta(geneExpr: float, epsilon: float, sigma: float, Kd: float) -> float:
    """
    Compute cofactor (ATH1)-dependent gene self-activation level based on Hill equation.

    Parameters
    ----------
    geneExpr : float64
        Gene expression quantity.
    epsilon : float64
        Model free parameter. Maximum self-activation level.
    sigma : float64
        Model free parameter. Hill coefficient.
    Kd : float64
        Model free parameter. Apparent dissociation constant.

    Returns
    -------
    theta : float64
        Cofactor-dependent gene self-activation level.
    """

    theta = epsilon * math.pow(geneExpr, sigma) / (Kd + math.pow(geneExpr, sigma))

    return theta


def calc_time_to_next_repl_after_ev(
    time_to_next_repl: np.ndarray[Any, float], cell_cycle: float, ev_time: float
) -> np.ndarray[Any, float]:
    """
    Calculate the time to next DNA replication after model evolution with constant cell cycle.

    Parameters
    ----------
    time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication in each trial before model evolution.
    cell_cycle : float64
        Cell cycle of the model.
    ev_time : float64
        Evolution time (unit: hour).

    Returns
    -------
    new_time_to_next_repl : NDAarray[float64], shape (N,)
        Time to next DNA replication after evolution.
    """

    if math.isinf(cell_cycle):
        new_time_to_next_repl = np.full(shape=time_to_next_repl.shape, fill_value=math.inf, dtype=np.float64)
    else:
        assert np.all(time_to_next_repl < cell_cycle)
        new_time_to_next_repl = (
            time_to_next_repl + np.ceil((ev_time - time_to_next_repl) / cell_cycle) * cell_cycle - ev_time
        )

    return new_time_to_next_repl


def calc_time_to_next_repl_after_ccc(
    old_time_to_next_repl: np.ndarray[Any, float], old_cell_cycle: float, new_cell_cycle: float
) -> np.ndarray[Any, float]:
    """
    Calculate the time to next DNA replication after cell cycle changes based on equal-scale transformation.
    This means that, if a cell has 1 hour to divide, and then cell cycle doubles, the transformed cell will
    divide 2 hours later.

    Parameters
    ----------
    old_time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication before cell cycle changes.
    ole_cell_cycle : float64
        Old cell cycle.
    new_cell_cycle : float64
        New cell cycle.

    Returns
    -------
    new_time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication after cell cycle changes.

    **NOTE** If cell cycle is infinite, it represents that cell stops dividing. If a cell restart division from
    quiescence, `new_time_to_next_repl` is set to 0. If `old_time_to_next_repl` is 0.0, the cell will immediately
    do a DNA replication and divide the next moment before new cycle is applied.
    """

    if math.isinf(old_cell_cycle):
        if math.isinf(new_cell_cycle):
            new_time_to_next_repl = np.full(shape=old_time_to_next_repl.shape, fill_value=math.inf)
        else:
            new_time_to_next_repl = np.full(shape=old_time_to_next_repl.shape, fill_value=0.0)
    else:
        assert np.all(old_time_to_next_repl < old_cell_cycle)

        # equal-scale transform
        # special treatment for NaN
        # if `old_time_to_next_repl` is 0.0 and `new_cell_cycle` is infinite, set `new_time_to_next_repl`  to 0.0
        transform_func = np.vectorize(pyfunc=lambda x, y, z: 0.0 if x == 0 and z == math.inf else x / y * z)
        new_time_to_next_repl = transform_func(old_time_to_next_repl, old_cell_cycle, new_cell_cycle)

    return new_time_to_next_repl


def calc_prot_fixed_points(
    Pme23: float,
    cell_cycle: float,
    cc0: float,
    mu: float,
    alpha_lim: float,
    epsilon: float,
    sigma: float,
    Kd: float,
    f_min: float,
    f_max: float,
    f_lim: float,
    Pt: float,
    gamma_transcr: float,
    prot_per_transcr: float,
    kappa: float,
    act_on_gene: bool = True,
    p0_array: Optional[np.ndarray] = None,#np.concatenate(([0], 10 ** np.arange(-4, 4, 0.1))),
    solver_prec: int = 32,
    verify_tol: float = 1e-16,
    ndigits: float = 8,
) -> np.ndarray[Any, float]:
    """
    Compute the fixed number of protein molecules for the production-degradation system to be stable or
    metastable (critical) at different chromatin methylation state.

    Parameters
    ----------
    Pme23 : float64
        Repressive methylation (me2/me3) modification level of STM. This parameter is confied in [0, 1].
    cell_cycle : float64
        Model free parameter. Cell cycle.
    cc0 : float64
        Model free parameter. Eeference cell cycle for normalization.
    mu : float64
        Model free parameter. Exponential factor.
    alpha_lim : float64
        Model free parameter. Lower bound of extended parameter `alpha`.
    epsilon : float64
        Model free parameter. Maximum effective activation level of local ATH1 protein.
    sigma : float64
        Model free parameter. Hill coefficient.
    Kd : float64
        Model free parameter. Dissociation constant.
    f_min : float64
        Model free parameter. Minimum transcription initiation rate.
    f_max : float64
        Model free parameter. Maximum transcription initiation rate.
    f_lim : float64
        Model free parameter. Upper bound of transcription initiation rate with activation.
    Pt : float64
        Model free parameter. Thereshold value of maximum repression.
    gamma_transcr : float64
        Model free parameter. Random transcription rate
    prot_per_transcr : float64
        Model free parameter. Average protein number to be translated per transcript.
    kappa : float64
        Model free parameter. Protein degradation rate.
    act_on_gene : bool (default=True)
        Whether to use major (True) or alternative (False) model.
    p0_array : NDArray[float64], shape (#,) (optional)
        Initial points (prediction root) to be used in sympy numerical solving.
    solver_prec : int32 (default=32)
        Precision (decimal places) of sympy solver.
    verify_tol : float64 (default=1e-20)
        Tolerance of deviation from 0 in verification of equation roots.
    ndigits : int32 (default=8)
        Decimal precision of roots.

    Returns
    -------
    prot_crtitical_points : NDArray[float64], shape (#,)
        Array of possible crtical points (ascending order).
    """

    assert 0 <= Pme23 <= 1, "Acceptable range of 'Pme23' is from 0 to 1."

    # define protein dynamic function
    p = sp.Symbol("p")  # protein number
    alpha = max((cc0 / cell_cycle) ** mu, alpha_lim) if act_on_gene else 1
    f_lin = f_max - min(Pme23 / Pt, 1) * (f_max - f_min)  # linear piecewise
    prot_dynamic_func = (
        prot_per_transcr * (alpha * epsilon * p**sigma / (Kd + p**sigma) * f_lin + gamma_transcr) - kappa * p
    )
    lim = alpha * epsilon * f_lin  # limit non-random transcription rate

    # define numerical solver
    def eq_solver(p0: float) -> float:
        """Get real and positive root for protein dynamic equation from a given initial point (p0)."""
        try:
            # by default, sympy solver will do verification while solving to only get reasonable root
            # however, that functions which are very steep near the root, the verification of the solution may fail
            # therefore, disable the built-in verification, use manual check instead
            root = complex(sp.nsolve(prot_dynamic_func, p, p0, verify=False, prec=solver_prec).evalf(chop=True))
            # manual check, keep positive real root only
            if root.imag == 0 and root.real >= 0:
                root = root.real
                if abs(prot_dynamic_func.subs(p, root)) > verify_tol:
                    root = math.nan  # mistake
            else:
                root = math.nan  # imaginary root or real but negative root
        except ValueError:
            root = math.nan  # if no root at all

        return root
    
    p0_array = np.concatenate(([0], 10 ** np.arange(-4, 4, 0.1))) if p0_array is None else p0_array
    
    # solve equation
    roots = np.vectorize(pyfunc=eq_solver, otypes=[np.float64])(p0_array)  # vectorize solver
    valid_roots = np.unique(np.round(roots[~np.isnan(roots)], decimals=ndigits))

    # check bound
    lim_root = prot_per_transcr * (f_lim + gamma_transcr) / kappa  # limit root (at limit transcription rate)
    if lim <= f_lim:
        # always below limit
        prot_critical_points = np.sort(valid_roots)
    else:
        # reach limit
        split_point = math.pow(Kd * (f_lim / lim) / (1 - f_lim / lim), 1 / sigma)  # where reach limit
        prot_critical_points = np.sort(valid_roots[valid_roots <= split_point])
        if lim_root > split_point:
            prot_critical_points = np.append(prot_critical_points, round(lim_root, ndigits=ndigits))

    return prot_critical_points
