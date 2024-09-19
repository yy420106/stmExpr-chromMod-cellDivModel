from typing import Any, List, Optional, Tuple, Union

import numba_progress as nbp
import numpy as np
from gillespie_ssa import gillespie_get_propensities, gillespie_ssa_parallel
from parameters import NUM_H3, DynamicParameters
from typing_extensions import Self
from util_funcs import (
    calc_alpha,
    calc_Pme23,
    calc_prot_fixed_points,
    calc_time_to_next_repl_after_ccc,
    calc_time_to_next_repl_after_ev,
)

__all__ = ["GeneChromModel"]


class GeneChromModel(DynamicParameters):
    """Gene expression-chromatin modification-cell division coupled model."""

    __H = NUM_H3  # private attribute

    def __init__(
        self,
        N: int = 0,
        geneExpr: Union[float, np.ndarray[Any, float], None] = None,
        meState: Optional[np.ndarray[Any, int]] = None,
        meState_fastBuild: Union[int, Tuple[int, ...]] = -1,
        time_to_next_repl: Union[float, np.ndarray[Any, float], None] = None,
        time: float = 0.0,
        act_on_gene: bool = True,
        **kwargs: float,
    ) -> None:
        """
        Constructor of class `model.GeneChromModel`.

        Parameters
        ----------
        N : int32 (default=0)
            If N > 0, it set the number of sub-models (monitored cells) to be integrated. If N = 0, just return
            a father container of class `parameters.DynamicParameters` and a few new attributes.
        geneExpr : float64 | NDArray[float64], shape (N,) (optional)
            Gene expression quantity (protein molecule number) in each sub-model. If a single float number is
            passed, it will be broadcasted to all sub-models. By default, it will be set randomly.
        meState : NDArray[int32], shape (H,) or (N, H) (optional)
            Methylation state of H3 histone at target gene locus in each sub-model. If a 1-D array is passed,
            it will be broadcasted to all sub-models. Note that this parameter has a higher priority than
            `meState_fastBuild` if both they are specified.
        meState_fastBuild : int32 | Tuple[int32, ...] (default=-1)
            Fast building method of chromatin methylation state. It should be a integer or a tuple of integer
            represents the building method for each sub-model, in the latter case, the tuple length must be N.
            Only -1, 0, 1, 2 or 3 is valid, where -1 use random setting and other k refers to uniformly set
            k-methylation. If a single integer is provided, it will be broadcasted to all sub-models.
        time_to_next_repl : float64 | NDArray[float64], shape (N,) (optional)
            Time interval (unit: hour) to next DNA replication (cell division) in each sub-model. By default,
            they are set randomly among all possibilities (discrete or continuous). Note that, 0 is also acceptable
            when cell cycle is infinte (division stop), because for this case `self.geneExpr` and `self.meState`
            refers to the model state at the very end of the last presumed cell cycle, this allows `self.meState` to
            be choose with more freedom. If a single float is provided, it will be broadcasted to all sub-models.
        time : float64 (default=0.0)
            Timestamp (unit: hour) that represents the objective time.
        act_on_gene : bool (default=True)
            Wheteher to use major (True) or alternative (False) model. See model description for details.
        **kwargs : float64
            Keyword arguments that specify the free parameters. See class `parameters.DynamicParameters` for details.

        Attributes
        ----------
        self.param_dict : Dict[str, float64] (inherited from class `parameters.DynamicParameters`)
        self.H : int32 (inherited from variable `parameters.NUM_H3`)
        self.N : int32
        self.geneExpr : NDArray[float64], shape (N,)
        self.meState : NDArray[int32], shape (N, H)
        self.time_to_next_repl : NDArray[float], shape (N,)
        self.time : float64
        self.act_on_gene : bool
        *self.all_param : Tuple[float64, ...]
            This attribute aims to provide a equivalent but simpler API as function `self.get_param("all")` do, 
            it is for query only and cannot be modified. See class `parameters.DynamicParamters` for more details.
        *self.prot_fp_param : Tuple[float64, ...]
            Extract parameters used for protein fixed points computation, equivalent to function `self.get_param("prot_fp")`.
        """

        # initialize class attributes
        super(GeneChromModel, self).__init__(**kwargs)  # build attributes 'param_dict'
        self.H = self.__H
        self.N = N
        self.geneExpr = None
        self.meState = None
        self.time_to_next_repl = None
        self.time = time
        self.act_on_gene = act_on_gene

        if N:
            # update attributes 'geneExpr'
            if geneExpr is None:
                # randomly set
                self.geneExpr = np.random.randint(low=0, high=100 * self.__H, size=(N,)).astype(np.float64)
            elif isinstance(geneExpr, np.ndarray):
                assert geneExpr.shape == (N,), "The shape of 'geneExpr' is not compatible."
                assert np.all(geneExpr >= 0), "Negative value is not acceptable for 'geneExpr'."
                self.geneExpr = geneExpr.astype(np.float64)
            else:
                assert geneExpr >= 0, "Negative value is not acceptable for 'geneExpr'."
                self.geneExpr = np.full(shape=(N,), fill_value=geneExpr, dtype=np.float64)

            # update attributes 'meState'
            if meState is None:
                if isinstance(meState_fastBuild, tuple):
                    assert len(meState_fastBuild) == N, "The length of 'meState_fastBuild' is not compatible."
                    # initialize
                    self.meState = np.empty(shape=(N, self.__H), dtype=np.int32)
                    for idx in range(N):
                        if meState_fastBuild[idx] == -1:
                            self.meState[idx] = np.random.randint(low=0, high=4, size=(self.__H,))
                        elif meState_fastBuild[idx] in [0, 1, 2, 3]:
                            self.meState[idx] = np.full(
                                shape=(self.__H,), fill_value=meState_fastBuild[idx], dtype=np.int32
                            )
                        else:
                            raise ValueError("Invalid option of 'meDtate_fastBuild' is provided.")
                else:
                    if meState_fastBuild == -1:
                        # randomly set
                        self.meState = np.random.randint(low=0, high=4, size=(N, self.__H))
                    elif meState_fastBuild in [0, 1, 2, 3]:
                        self.meState = np.full(shape=(N, self.__H), fill_value=meState_fastBuild, dtype=np.int32)
                    else:
                        raise ValueError("Invalid option of 'meDtate_fastBuild' is provided.")
            else:
                assert 1 <= meState.ndim <= 2, "Only 1-D or 2-D array is acceptable for 'meState'."
                assert np.isin(meState, [0, 1, 2, 3]).all(), "Invalid methylation state in 'meState' is provided."
                if meState.ndim == 2:
                    assert meState.shape == (N, self.__H), "The shape of 'meState' is not compatible."
                    self.meState = meState.astype(np.int32)
                else:
                    assert meState.shape == (self.__H,), "The shape of 'meState' is not compatible."
                    self.meState = np.tile(meState.astype(np.int32), reps=(N, 1))

            # update attributes 'time_to_next_repl'
            if time_to_next_repl is None:
                # randomly set
                if self.param_dict["cell_cycle"] == np.inf:
                    self.time_to_next_repl = np.random.choice((0.0, np.inf), size=N, replace=True)
                else:
                    self.time_to_next_repl = np.random.uniform(0.0, self.param_dict["cell_cycle"], size=N)
            else:
                if isinstance(time_to_next_repl, np.ndarray):
                    assert time_to_next_repl.shape == (N,), "The shape of 'time_to_next_repl' is not compatible."
                    self.time_to_next_repl = time_to_next_repl.astype(np.float64)
                else:
                    self.time_to_next_repl = np.full(shape=(N,), fill_value=time_to_next_repl, dtype=np.float64)

                # check time compatibility
                if self.param_dict["cell_cycle"] == np.inf:
                    assert np.all(
                        (self.time_to_next_repl == np.inf) | (self.time_to_next_repl == 0.0)
                    ), "Logical problem: 'time_ro_next_repl' and 'cell_cycle' is not compatible."
                else:
                    assert (
                        0 <= self.time_to_next_repl.min()
                        and self.time_to_next_repl.max() < self.param_dict["cell_cycle"]
                    ), "Logical problem: 'time_ro_next_repl' and 'cell_cycle' is not compatible."
        else:
            pass

    @property
    def all_param(self) -> Tuple[float, ...]:
        return super().get_param(param_names="all")

    @property
    def prot_fp_param(self) -> Tuple[float, ...]:
        return super().get_param(param_names="prot_fp")

    def set_free_param(self, **kwargs: float) -> None:
        """
        Set values of model free parameters.

        Parameters
        ----------
        **kwargs: float64
            Keyword arguments that specify the free model parameters to their new values.

        **NOTE** This function is a rewrite of the father class method and makes some extensions. If free parameter
        `cell_cycle` is to be changed, then an additional modification of model attribute `self.time_to_next_repl`
        will be done accordingly because these two values are closely correlated. See function
        `util_funcs.calc_time_to_next_repl_after_ccc` for more details.
        """

        try:
            # check if cell cycle is changed
            new_cell_cycle = kwargs["cell_cycle"]
            old_cell_cycle = self.param_dict["cell_cycle"]
            self.time_to_next_repl = calc_time_to_next_repl_after_ccc(
                old_time_to_next_repl=self.time_to_next_repl,
                old_cell_cycle=old_cell_cycle,
                new_cell_cycle=new_cell_cycle,
            )  # update time to next replication
        except KeyError:
            pass

        super().set_free_param(**kwargs)  # update free parameters

    def get_propensities(
        self, buffer: bool = True, alpha0: float = None, time_delta: float = None
    ) -> List[Tuple[np.ndarray[Any, float], np.ndarray[Any, float], float, float]]:
        """
        Compute current propensities of 4 possible event: H3 methylation, H3 demethylation, gene transcription &
        protein degreadtion for each sample. See function `gillespie_ssa.gillespie_get_propensities` for more details.
        
        Return a list containing N tuples, where each tuple represents a sample and composed of 4 elements, like
        (mePropensity_i, demPropensity_i, exprPropensity_i, pdgrPropensity_i), for i = 1,2,...,N.
        """
        return [
            gillespie_get_propensities(
                geneExpr=self.geneExpr[i],
                meState=self.meState[i],
                all_param=self.all_param,
                act_on_gene=self.act_on_gene,
                buffer=buffer,
                buffer_ref=(
                    calc_alpha(*self.get_param(param_names="alphaT")) if alpha0 is None else alpha0,
                    self.time if time_delta is None else time_delta,
                ),
            )
            for i in range(self.N)
        ]

    def get_fixed_points(self) -> List[np.ndarray[Any, float]]:
        """
        Compute the fixed number of protein molecules for each sample. Return a list containing N arrays.
        See function `util_funcs.calc_prot_fixed_points` for more details.
        """
        return [
            calc_prot_fixed_points(calc_Pme23(self.meState[i]), *self.prot_fp_param, act_on_gene=self.act_on_gene)
            for i in range(self.N)
        ]

    def evolve(
        self,
        ev_time: float,
        time_step: float,
        buffer: bool = True,
        alpha0: Optional[float] = None,
        p_bar: Optional[nbp.ProgressBar] = None,
        sizeFactor: float = 1.1,
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, int], np.ndarray[Any, float], float]:
        """
        Update model over time and record the intermediate state during evolution.

        Parameters
        ----------
        ev_time : float64
            Evolution time (unit: hour). Non-negative (include 0) is acceptable.
        time_step : float64
            Record time step.
        buffer : bool (default=True)
            Determine whether to use buffering strategy for cycle-dependent activation level `alpha`.
        alpha0 : float (optional)
            Reference value for buffering `alpha`.
        p_bar : ProgressBar (optional)
            A numba implementation object of tqdm to show the progress.
        sizeFactor : float64 (default=1.1)
            Factor that controls the initial array size of `samples_transcrT`. See **NOTE**.

        Returns
        -------
        time_records : NDArray[float64], shape (T,)
            Array of time points (unit: hour) at which to monitor the model state.
        samples_geneExpr : NDArray[float64], shape (N, T)
            Records of gene expression quantity (protein level) during evolution.
        samples_meState : NDArray[int32], shape (N, T, H)
            Records of the methylation state of histone during evolution.
        samples_transcrT : NDArray[float64], shape (N, #)
            Array that store the time of every transcription event in each trial, empty spaces are filled with
            NaN. "#" indicates the length is non-constant.
        alpha_end : float
            When buffer strategy is used, return the value of `alpha` at the end of simulation. This is useful
            when simulation stops in buffering stage before `alpha` reach its target value.

        **NOTE** If out-of-bounds (index error) occurs, consider increasing parameter `sizeFactor`. See function
        `gillespie_ssa.gillespie_ssa_parallel` for more details. Usually the default value (1.1) is enough.
        """

        # set record time points
        if ev_time > 0:
            n = ev_time / time_step
            time_records = np.linspace(self.time, self.time + int(n) * time_step, int(n) + 1)
            if n != int(n):
                time_records = np.append(time_records, self.time + ev_time)  # add the end time
        elif ev_time == 0:
            time_records = np.array([self.time])
        else:
            raise ValueError("'ev_time' must be non-negative.")

        # evolution & sampling
        samples_geneExpr, samples_meState, samples_transcrT, alpha_end = gillespie_ssa_parallel(
            geneExpr0=self.geneExpr,
            meState0=self.meState,
            time_to_next_repl0=self.time_to_next_repl,
            all_param=self.all_param,
            time_records=time_records,
            act_on_gene=self.act_on_gene,
            buffer=buffer,
            alpha0=calc_alpha(*self.get_param(param_names="alphaT")) if alpha0 is None else alpha0,
            p_bar=p_bar,
            sizeFactor=sizeFactor,
        )

        # update model state to the final time
        self.geneExpr = samples_geneExpr[:, -1]
        self.meState = samples_meState[:, -1, :]
        self.time_to_next_repl = calc_time_to_next_repl_after_ev(
            time_to_next_repl=self.time_to_next_repl,
            cell_cycle=self.param_dict["cell_cycle"],
            ev_time=ev_time,
        )
        self.time += ev_time

        return time_records, samples_geneExpr, samples_meState, samples_transcrT, alpha_end

    def extract_sub_model(self, indices_or_condition: np.ndarray[Any, int | bool]) -> Self:
        """Extract sub-model from a existed `model.GeneChromModel` object."""
        return self.__class__.__extract_sub_model(model=self, indices_or_condition=indices_or_condition)

    @classmethod
    def __extract_sub_model(cls, model: Self, indices_or_condition: np.ndarray[Any, int | bool]) -> Self:
        """
        Extract sub-model from a existed 'model.GeneChromModel' object. (private method)

        Parameters
        ----------
        model : GeneChromModel
            The father model to be extracted.
        indices_or_condition : NDArray[int | bool], shape (#,)
            Indices (integer array) of sub-model or a filter condition (bool array) that specify the samples to
            be extracted.

        Returns
        -------
        sub_model : GeneChromModel
            The sub-model that contains those single sample which satisfy the condition or with the queried indices.
        """

        assert isinstance(indices_or_condition, np.ndarray), "'indices_or_condition' must be a numpy.array."

        # save parameters
        kwargs = {key: model.param_dict[key] for key in model.default_free_param_dict().keys()}

        if indices_or_condition.dtype == int:
            # indices
            sub_model = cls(
                N=len(indices_or_condition),
                geneExpr=model.geneExpr.take(indices=indices_or_condition, axis=0),
                meState=model.meState.take(indices=indices_or_condition, axis=0),
                time_to_next_repl=model.time_to_next_repl.take(indices=indices_or_condition, axis=0),
                time=model.time,
                **kwargs,
            )
        elif indices_or_condition.dtype == bool:
            # condition
            sub_model = cls(
                N=indices_or_condition.sum(),
                geneExpr=model.geneExpr[indices_or_condition],
                meState=model.meState[indices_or_condition],
                time_to_next_repl=model.time_to_next_repl[indices_or_condition],
                time=model.time,
                **kwargs,
            )
        else:
            raise TypeError("Only 'int' or 'bool' is acceptable data type for array 'indices_or_condition'.")

        return sub_model
