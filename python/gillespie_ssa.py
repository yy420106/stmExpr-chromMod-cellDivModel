import math
from typing import Any, Optional, Tuple

import numba as nb
import numba_progress as nbp
import numpy as np
from util_funcs import calc_alpha, calc_E, calc_omega, calc_Pme23, calc_theta

__all__ = ["gillespie_get_propensities", "gillespie_draw", "gillespie_ssa", "gillespie_ssa_parallel"]  # JIT-complie


@nb.njit(
    signature_or_function=nb.types.Tuple(types=(nb.float64[:], nb.float64[:], nb.float64, nb.float64))(
        nb.float64,
        nb.int32[:],
        nb.types.UniTuple(dtype=nb.float64, count=31),
        nb.boolean,
        nb.boolean,
        nb.types.UniTuple(dtype=nb.float64, count=2),
    ),
    cache=True,
)
def gillespie_get_propensities(
    geneExpr: float,
    meState: np.ndarray[Any, int],
    all_param: Tuple[float, ...],
    act_on_gene: bool,
    buffer: bool,
    buffer_ref: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], float, float]:
    """
    Compute propensities of H3 methylation, H3 demethylation, gene transcription and protein degradation
    based on the current gene expression, chromatin methylation state and model parameters.

    Parameters
    ----------
    geneExpr : float64
        Gene expression quantity.
    meState : NAarray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.
    all_param : Tuple[float64, ...]
        A 31-element tuple specify the model parameters value, in the order of `cell_cycle`, `cc0`, `mu`, `alpha_lim`,
        `alpha_expk`, `epsilon`, `sigma`, `Kd`, `f_min`, `f_max`, `f_lim`, `Pt`, `gamma_transcr`, `prot_per_transcr`,
        `kappa`, `beta`, `e_distal`, `rho`, `k_me`, `Pdem`, `Pex`, `A`, `B`, `omega_lim`, `k_me01`, `k_me12`, `k_me23`,
        `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`. See class `parameters.DynamicParameters` for more details.
    act_on_gene : bool
        Whether to apply the major model (directly act on gene transcription) or alternative model (indirectly act
        on gene transcription by epigentic modification).
    buffer : bool
        Determine whether to use buffering strategy for cycle-dependent activation level `alpha`.
    buffer_ref : Tuple[float, float]
        Reference information used for computing buffer value of `alpha`, in the order of `alpha0`, `time_delta`,
        where `alpha0` is reference value, `time_delta` is time increment. This parameter is used only if `buffer`
        set to True.

    Returns
    -------
    mePropensity : NDArray[float64], shape (H,)
        Methylation propensity for each H3 histone.
    demPropensity : NDArray[float64], shape (H,)
        Demethylation propensity for each H3 histone.
    exprPropensity : float64
        Gene transcription propensity.
    pdgrPropensity : float64
        Protein degradation propensity.
    """

    # Unpack parameters
    (
        cell_cycle,
        cc0,
        mu,
        alpha_lim,
        alpha_expk,
        epsilon,
        sigma,
        Kd,
        f_min,
        f_max,
        f_lim,
        Pt,
        gamma_transcr,
        _,
        kappa,
        beta,
        e_distal,
        rho,
        _,
        _,
        _,
        A,
        B,
        omega_lim,
        k_me01,
        k_me12,
        k_me23,
        gamma_me01,
        gamma_me12,
        gamma_me23,
        gamma_dem,
    ) = all_param

    # intermediate variable
    Pme23 = calc_Pme23(meState=meState)  # inhibtory methylation ratio
    E = calc_E(meState=meState, rho=rho, e_distal=e_distal)  # neighbor enhancement
    theta = calc_theta(geneExpr=geneExpr, epsilon=epsilon, sigma=sigma, Kd=Kd)  # cofactor-dependent activation level

    if act_on_gene:
        # cycle-dependent activation level
        alpha, omega = calc_alpha(cell_cycle=cell_cycle, cc0=cc0, mu=mu, alpha_lim=alpha_lim), 1.0

        # buffer alpha if needed
        if buffer:
            alpha0, time_delta = buffer_ref
            if alpha0 <= alpha:
                alpha = min(alpha0 * math.exp(alpha_expk * time_delta), alpha)
            else:
                alpha = max(alpha0 * math.exp(-alpha_expk * time_delta), alpha)
    else:
        alpha, omega = 1.0, calc_omega(cell_cycle=cell_cycle, A=A, B=B, omega_lim=omega_lim)

    # compute propensities
    mePropensity = (
        omega
        * beta
        * (
            (gamma_me01 + k_me01 * E) * (meState == 0)
            + (gamma_me12 + k_me12 * E) * (meState == 1)
            + (gamma_me23 + k_me23 * E) * (meState == 2)
        )
    )
    demPropensity = gamma_dem * (meState > 0)
    exprPropensity = min(alpha * theta * (f_max - min(Pme23 / Pt, 1) * (f_max - f_min)), f_lim) + gamma_transcr
    pdgrPropensity = geneExpr * kappa

    return mePropensity, demPropensity, exprPropensity, pdgrPropensity


@nb.njit(
    signature_or_function=nb.types.Tuple(types=(nb.float64, nb.int32, nb.int32))(
        nb.float64,
        nb.int32[:],
        nb.types.UniTuple(dtype=nb.float64, count=31),
        nb.boolean,
        nb.boolean,
        nb.types.UniTuple(dtype=nb.float64, count=2),
    ),
    cache=True,
)
def gillespie_draw(
    geneExpr: float,
    meState: np.ndarray[Any, int],
    all_param: Tuple[float, ...],
    act_on_gene: bool,
    buffer: bool,
    buffer_ref: Optional[Tuple[float, float]],
) -> Tuple[float, int, int]:
    """
    Draws a event and the time it took to do that event in a poisson process.

    Parameters
    ----------
    geneExpr : float64
    meState : NDArray[int32], shape (H,)
    all_param : Tuple[float64, ...]
    act_on_gene : bool
    buffer : bool
    buffer_ref : Tuple[float, float]

    NOTE: See function `gillespie_ssa.gillespie_get_propensities` for more details about these parameters.

    Returns
    -------
    deltaT : float64
        Time interval (unit: sec) for next comming event.
    event_class : int32 (in 0, 1, 2, 3)
        Next event marks, with 0, 1, 2, 3 represents H3 methylation, H3 demethylation, gene transcription
        and protein degradation, respectively.
    histone_idx : int32 (in 0, 1, 2, ... , H-1)
        Histone index in which next event to be occured. Note that this value is useful only if methylation
        or demethylation occurs, otherwise it is set to H3 histone number (H) and meaningless.
    """

    # compute propensity distribution
    mePropensity, demPropensity, exprPropensity, pdgrPropensity = gillespie_get_propensities(
        geneExpr=geneExpr,
        meState=meState,
        all_param=all_param,
        act_on_gene=act_on_gene,
        buffer=buffer,
        buffer_ref=buffer_ref,
    )
    props = np.append(arr=np.concatenate((mePropensity, demPropensity)), values=[exprPropensity, pdgrPropensity])
    props_sum = props.sum()

    # compute next time
    # Principles
    # The interval time distribution of the Poisson process is exponential
    # X1 ~ Exp(k1), X2 ~ Exp(k2), then min(X1, X2) ~ Exp(k1 + k2)
    deltaT = np.random.exponential(scale=1.0 / props_sum)

    # draw event from this distribution
    q = np.random.rand() * props_sum
    idx = 0
    p_sum = 0.0
    while p_sum <= q:
        p_sum += props[idx]
        idx += 1

    # classify event
    if idx <= 2 * meState.size:
        event_class, histone_idx = (idx - 1) // meState.size, (idx - 1) % meState.size
    else:
        event_class, histone_idx = 3 + idx - props.size, meState.size

    return deltaT, event_class, histone_idx


@nb.njit(boundscheck=True, nogil=True, cache=True)
def gillespie_ssa(
    geneExpr0: float,
    meState0: np.ndarray[Any, int],
    time_to_next_repl0: float,
    all_param: Tuple[float, ...],
    time_records: np.ndarray[Any, float],
    act_on_gene: bool,
    buffer: bool,
    alpha0: float,
    sizeFactor: float = 1.1,
    calc_alpha_end: bool = True,
) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, int], np.ndarray[Any, float], float]:
    """
    Gillespie stochastic simulation algorithm (SSA).

    Parameters
    ----------
    geneExpr0 : float64
        Initial gene expression quantity at the start of simulation.
    meState0 : NDArray[int32], shape (H,)
        Initial methylation state of each H3 histone at the start of simulation.
    time_to_next_repl0 : float64
        Time interval (unit: hour) between simulation initiation and the 1st DNA replication after that. Note that
        if it set to 0.0 (in most case), then `geneExpr0` and `meState0` refers to the model state at the very end
        of last cell cycle, and a cell division will do immediately.
    all_param : Tuple[float64, ...]
        A 31-element tuple specify the model parameters value, in the order of `cell_cycle`, `cc0`, `mu`, `alpha_lim`,
        `alpha_expk`, `epsilon`, `sigma`, `Kd`, `f_min`, `f_max`, `f_lim`, `Pt`, `gamma_transcr`, `prot_per_transcr`,
        `kappa`, `beta`, `e_distal`, `rho`, `k_me`, `Pdem`, `Pex`, `A`, `B`, `omega_lim`, `k_me01`, `k_me12`, `k_me23`,
        `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`. See class `parameters.DynamicParameters` for more details.
    time_records : NDArray[float64], shape (T,)
        Array of time points (unit: hour) at which to monitor the model state.
    act_on_gene : bool
        Whether to apply the major model (directly act on gene transcription) or alternative model (indirectly act
        on gene transcription by epigentic modification).
    buffer : bool
        Determine whether to use buffering strategy for cycle-dependent activation level `alpha`.
    alpha0 : float64
        Reference value for buffering `alpha`.
    sizeFactor : float64 (default=1.1)
        Factor that controls the initial array size of return `transcrT_records`. See **NOTE**.
    calc_alpha_end : bool (default=True)
        Determine whether to compute `alpha` value at the end of simulation. This parameter is recommanded to set
        to False only if you want to avoid redundant work in parallel computing (see function
        `gillespie_ssa.gillespie_ssa_parallel`) or you are sure that simulation time is long enough for `alpha` to
        buffer completely, otherwise it should be always set to True by default.

    Returns
    -------
    geneExpr_records : NDArray[float64], shape (T,)
        1-D array, with entry t is the gene expression quantity at time_records[t]
    meState_records : NDArray[int32], shape (T, H)
        2-D array, with entry (t, h) is the methylation state of histone with index h at time_records[t]
    transcrT_records : NDArray[float64], shape (#,)
        1-D array with enough length that store the time of every transcription event, empty spaces are filled with
        NaN. "#" indicates the length is nontrivial, which correlates with function parameter `sizeFactor`,
        `time_records` and model parameter `f_lim`.
    alpha_end : float
        When buffer strategy is used, return the value of `alpha` at the end of simulation. This is useful when
        simulation stops in buffering stage before `alpha` reach its target value.


    **NOTE** By default, Numba will not do bounds checking (a C/C++ feature) after compilation. However, if bounds
    checking is disabled, out of bounds accesses sometimes can produce garbage results, segfaults or crash.

    In this function, the size of array `transcrT_records` (adjusted by `sizeFactor`) is pre-allocate, because it
    is always better than to dynamically adjust it every time. To ensure that the array `transcrT_records` array is
    long enough to store all transcription time, program always need to check if subscript are out of bounds, but
    it will add extra work and decrease the execution speed and eficiency. On the other hand, if `sizeFactor` is too
    big, initialize a long array also cause time waste.

    There are 3 possible solutions to enable bounds checking:

    Option 1. Numba environment variable.
        Set the `NUMBA_BOUNDSCHECK` environment variable to 0 or 1 ahead to globally override this flag.

            >>> import os
            >>> os.environ["NUMBA_BOUNDSCHECK"] = "1"

    Option 2. Numba keywords `boundscheck`.
        Set `boundscheck` to True in JIT decorator, out of bounds accesses will raise `IndexError`.

            >>> import numba as nb
            >>> @nb.njit(boundscheck=True)

    Option 3. Manually check.
        Use raise sentence in JIT-decorated function to produce IndexError manually if out-of-bound occurs.

            >>> ... raise IndexError("Index is out of bounds.")

    Note that it is recommended (but not necessary) to only use bounds checking flag in debugging. After picking a
    suitable value of `sizeFactor`, e.g. 1.1 (default), cancel bounds checking in production simulation to get high
    performance.
    """

    # extract some useful parameters
    cell_cycle, cc0, mu, alpha_lim, alpha_expk = all_param[:5]
    f_lim = all_param[10]
    prot_per_transcr = all_param[13]
    Pdem = all_param[19]
    Pex = all_param[20]

    # initialize output
    geneExpr_records = np.empty(shape=(time_records.size,), dtype=np.float64)
    geneExpr_records[0] = geneExpr0

    meState_records = np.empty(shape=(time_records.size, meState0.size), dtype=np.int32)
    meState_records[0, :] = meState0

    transcrT_records = np.full(
        shape=(int(sizeFactor * (time_records[-1] - time_records[0]) * 60 * 60 * f_lim),),
        fill_value=np.nan,
        dtype=np.float64,
    )  # initialize a huge array to store transcription time; increase 'sizeFactor' value if IndexError is raised
    transcr_idx = 0

    # temporary variables
    curr_geneExpr = geneExpr0  # current gene expression
    curr_meState = meState0.copy()  # current methylation state
    curr_time = time_records[0]  # current time [hour]
    next_time_records_idx = 1  # next time index at which gene expression and methylation state need to be recorded
    next_repl_time = time_records[0] + time_to_next_repl0  # next DNA replication time [hour]

    # evolution loop
    while next_time_records_idx < time_records.size:
        while curr_time < time_records[next_time_records_idx]:
            # draw the event and interval time
            deltaT, event_class, histone_idx = gillespie_draw(
                geneExpr=curr_geneExpr,
                meState=curr_meState,
                all_param=all_param,
                act_on_gene=act_on_gene,
                buffer=buffer,
                buffer_ref=(alpha0, curr_time - time_records[0]),
            )
            delta_time = deltaT / (60 * 60)  # convert unit from second to hour

            # save current model state before evolution
            prev_geneExpr = curr_geneExpr
            prev_meState = curr_meState.copy()

            if curr_time + delta_time < next_repl_time:
                curr_time += delta_time  # update time

                # update current methylation state
                if event_class == 0:
                    curr_meState[histone_idx] += 1  # H3 methylation
                elif event_class == 1:
                    curr_meState[histone_idx] -= 1  # H3 demethylation
                elif event_class == 2:
                    curr_geneExpr += prot_per_transcr  # gene expression

                    # transcription-coupled demethylation
                    curr_meState[(np.random.rand(curr_meState.size) < Pdem) & (curr_meState > 0)] -= 1

                    # transcription-coupled histone exchange
                    nucleosome_idx = np.argwhere(np.random.rand(curr_meState.size // 2) > (1 - Pex) ** 2).ravel()
                    curr_meState[2 * nucleosome_idx] = 0
                    curr_meState[2 * nucleosome_idx + 1] = 0

                    # record transcription time
                    transcrT_records[transcr_idx] = curr_time
                    transcr_idx += 1
                else:
                    curr_geneExpr = max(curr_geneExpr - 1, 0.0)  # protein degradation (filter negative value)
            else:
                # reset time when meeting cell cycle
                curr_time = next_repl_time

                # nucleosomes reassemble after DNA-replication
                nucleosome_idx = np.argwhere(np.random.rand(curr_meState.size // 2) > 0.5).ravel()
                curr_meState[2 * nucleosome_idx] = 0
                curr_meState[2 * nucleosome_idx + 1] = 0

                # update next replication time
                next_repl_time += cell_cycle

        # update methylation state from next recording time to current time
        temp_idx = np.searchsorted(time_records > curr_time, True)  # the first time index after current time
        geneExpr_records[next_time_records_idx:temp_idx] = prev_geneExpr
        meState_records[next_time_records_idx:temp_idx] = prev_meState

        # update next recording time index
        next_time_records_idx = temp_idx

    # compute alpha at the end of simulation
    alpha = calc_alpha(
        cell_cycle=cell_cycle, cc0=cc0, mu=mu, alpha_lim=alpha_lim
    )  # stable cycle-dependent activation level

    if buffer and calc_alpha_end:
        if alpha0 <= alpha:
            alpha_end = min(alpha0 * math.exp(alpha_expk * (time_records[-1] - time_records[0])), alpha)
        else:
            alpha_end = max(alpha0 * math.exp(-alpha_expk * (time_records[-1] - time_records[0])), alpha)
    else:
        alpha_end = alpha

    return geneExpr_records, meState_records, transcrT_records, alpha_end


@nb.njit(parallel=True, nogil=True, cache=True)
def gillespie_ssa_parallel(
    geneExpr0: np.ndarray[Any, float],
    meState0: np.ndarray[Any, int],
    time_to_next_repl0: np.ndarray[Any, float],
    all_param: Tuple[float, ...],
    time_records: np.ndarray[Any, float],
    act_on_gene: bool,
    buffer: bool,
    alpha0: float,
    p_bar: Optional[nbp.ProgressBar] = None,
    sizeFactor: float = 1.1,
) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, int], np.ndarray[Any, float], float]:
    """
    Multiple samples parallel version of function `gillespie_ssa.gillespie_ssa`, each sample is an independent
    parallel computing branch. This function provide a compatible API for class `model.GeneChromModel`.

    Parameters
    ----------
    geneExpr0 : NDArray[floa64], shape (N,)
        Initial gene expression quantity of each trial at the start of simulation, its length is the number of
        total trials (N).
    meState0 : NDArray[int32], shape (N, H)
        Initial methylation state of chromatin H3 histone of each trial at the start of simulation. Axis 0
        represents trial, axis 1 represents histone.
    time_to_next_repl0 : NDArray[float64], shape (N,)
        Time interval (unit: hour) between the simulation initiation time and the 1st DNA replication after that
        of each trial.
    all_param : Tuple[float64, ...]
    time_records : NDArray[float64], shape (T,)
    act_on_gene : bool
    buffer : bool
    alpha0 : float
    p_bar : ProgressBar (optional)
        A numba implementation object of tqdm to show the progress.
    sizeFactor : float64 (default=1.1)

    **NOTE** `geneExpr0`, `meState0` and `time_to_next_repl0` must have the same size in the 1st dimension. See
    function `gillespie_ssa.gillespie_ssa` for more details about these parameters.

    Returns
    -------
    samples_geneExpr : NDArray[float64], shape (N, T)
        2-D array, with entry (n, t) is the gene expression quantity at time_records[t] in trial[n].
    samples_meState : NDArray[int32], shape (N, T, H)
        3-D array, with entry (n, t, h) is the methylation state of histone with index h at time_records[t]
        in trial[n].
    samples_transcrT : NDArray[float64], shape (N, #)
        2-D array with enough length that store the time of every transcription event in each trial, empty spaces
        are filled with NaN. '#' indicates the length is nontrivial, see function 'gillespie_ssa.gillespie_ssa'
        for more details.
    alpha_end : float
    """

    # check
    assert geneExpr0.size == meState0.shape[0] and meState0.shape[0] == time_to_next_repl0.size
    N, H = meState0.shape

    # extract some useful parameters
    cell_cycle, cc0, mu, alpha_lim, alpha_expk = all_param[:5]
    f_lim = all_param[10]

    # initialize output
    samples_geneExpr = np.empty(shape=(N, time_records.size), dtype=np.float64)
    samples_meState = np.empty(shape=(N, time_records.size, H), dtype=np.int32)
    samples_transcrT = np.empty(
        shape=(N, int(sizeFactor * (time_records[-1] - time_records[0]) * 60 * 60 * f_lim)), dtype=np.float64
    )

    # parallel loop
    for n in nb.prange(N):
        samples_geneExpr[n], samples_meState[n], samples_transcrT[n], _ = gillespie_ssa(
            geneExpr0=geneExpr0[n],
            meState0=meState0[n],
            time_to_next_repl0=time_to_next_repl0[n],
            all_param=all_param,
            time_records=time_records,
            act_on_gene=act_on_gene,
            buffer=buffer,
            alpha0=alpha0,
            sizeFactor=sizeFactor,
            calc_alpha_end=False,
        )
        if p_bar is not None:
            p_bar.update(1)  # update

    # compute alpha at the end of simulation
    alpha = calc_alpha(
        cell_cycle=cell_cycle, cc0=cc0, mu=mu, alpha_lim=alpha_lim
    )  # stable cycle-dependent activation level

    if buffer:
        if alpha0 <= alpha:
            alpha_end = min(alpha0 * math.exp(alpha_expk * (time_records[-1] - time_records[0])), alpha)
        else:
            alpha_end = max(alpha0 * math.exp(-alpha_expk * (time_records[-1] - time_records[0])), alpha)
    else:
        alpha_end = alpha

    return samples_geneExpr, samples_meState, samples_transcrT, alpha_end
