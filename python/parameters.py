import copy
import math
from collections import OrderedDict
from operator import itemgetter
from typing import List, Tuple, Union

__all__ = ["GENE_LEN", "NUM_H3", "load_default_free_param_dict", "update_depend_param", "DynamicParameters"]

"""
Model Parameters Explanation
============================>

Structure Overview
------------------
Parameters ----- Static (#2)
             |
             --- Dynamic (#31) ----- Free (#24) ----- Major (#21)
                                 |                |
                                 |                --- Alternative (#3, marked with **)
                                 |
                                 --- Dependent (#7, marked with *)

Static (global/fixed)
---------------------
GENE_LEN : length (3482 bp) of STM gene locus.
NUM_H3 : number of H3 histones at STM gene locus base on the evaluation that one nucleosome unit contains 
        ~200 bp. Histones with indices (start from 0) 2i and 2i+1 belong to the same nucleosome.

Dynamic (variable)
------------------
1. Cell division
    cell_cycle : cell cyle [hour]

2. Gene activation
    cc0 : normalized cell cycle [hour]
    mu : exponential of cycle-dependent STM gene activation level relative to cell divison frequency
    alpha_lim : lower bound of cycle-dependent STM gene activation level
    alpha_expk : exponential factor of cycle-dependent STM gene activation level in buffering time [1/hour]
    epsilon : maximum self-activation level of STM
    sigma : Hill coefficient
    Kd : apparent dissociation constant of ATH1-STM complex

3. Gene transcription
    f_min : minimum transcription initiation rate [1/sec]
    f_max : maximum transcription initiation rate [1/sec]
    f_lim : upper bound of transcription initiation rate when considering activation [1/sec]
    Pt : Threshold proportion of H3K27me3/me2 marks at which maximal repression reaches
    gamma_transcr : random transcription rate [1/sec]

4. Protein production & degradation
    prot_per_transcr : average protein number to be translated per transcript
    kappa : protein degradation rate [1/sec]

5. Histone methylation
    beta : local PRC2 activity (affected by PRC2 elements recruiter, e.g. RBE/BPC in this research)
    e_distal : distal gain contribution
    rho : activation level of PRC2 by 2-methylation relative to 3-methylation
    k_me : reference PRC2-mediated methylation rate [1/sec]
    *k_me01 : PRC2-mediated 1-methylation rate {9*k_me} [1/sec]
    *k_me12 : PRC2-mediated 2-methylation rate {6*k_me} [1/sec]
    *k_me23 : PRC2-mediated 3-methylation rate {k_me} [1/sec]
    *gamma_me01 : random 1-methylation rate by noise {k_me01/20} [1/sec]
    *gamma_me12 : random 2-methylation rate by noise {k_me12/20} [1/sec]
    *gamma_me23 : random 3-methylation rate by noise {k_me23/20} [1/sec]

6. Histone demethylation
    Pdem : transcription-coupled demethylation probability
    Pex : transcription-coupled histone exchange probability
    *gamma_dem : integrated random demethylation rate {fMin*Pdem} [1/sec] 

7. Supplementary (used in alternative model)
    **A : Absolute of slope on exponential [1/hour]
    **B : Intercept on exponential
    **omega_lim : Upper bound of cycle-dependent activation level on PRC2

Notes & Hints
-------------
1. Square brackets show units; variable start with * only means we set its default value in this model 
    depends on other parameters (but we still refer them as dynamic parameters instead of intermediate 
    variables because they still have independent physical meanings), the formula is shown in braces.

2. The symbols of some parameters in the program (left) may different from those in the paper (right): 
    e.g. 'cell_cycle' -> 'T', 'cc0' -> 'T0', 'alpha_expk' -> 'F', 'prot_per_transcr' -> 'n_ppt'.
<============================
"""

# static parameter (fixed)
GENE_LEN = 3482  # bp
NUM_H3 = 2 * math.ceil(GENE_LEN / 200)  # H3 histone number


def load_default_free_param_dict() -> OrderedDict[str, float]:
    """
    Load free parameters with their default values.

    Returns
    -------
    free_param_dict : Dict[str, float64]
        Dictionary that maps model free parameters to their default values.
    """

    free_param_dict = OrderedDict()

    # cell division
    free_param_dict["cell_cycle"] = 22.0
    # gene activation
    free_param_dict["cc0"] = 22.0
    free_param_dict["mu"] = 0.5
    free_param_dict["alpha_lim"] = 1e-2
    free_param_dict["alpha_expk"] = math.log(2) / 22.0
    free_param_dict["epsilon"] = 1.0
    free_param_dict["sigma"] = 2.0
    free_param_dict["Kd"] = 180.0
    # gene transcription
    free_param_dict["f_min"] = 1e-4
    free_param_dict["f_max"] = 4e-3
    free_param_dict["f_lim"] = 1 / 60
    free_param_dict["Pt"] = 1 / 3
    free_param_dict["gamma_transcr"] = 9e-8
    # protein production & degradation
    free_param_dict["prot_per_transcr"] = 1.0
    free_param_dict["kappa"] = 4e-6
    # histone methylation
    free_param_dict["beta"] = 1.0
    free_param_dict["e_distal"] = 0.001
    free_param_dict["rho"] = 1 / 10
    free_param_dict["k_me"] = 8e-6
    # histone demethylation & exchange
    free_param_dict["Pdem"] = 5e-3
    free_param_dict["Pex"] = 1.5e-3
    # supplementary (used in alternative model)
    free_param_dict["A"] = 0.5
    free_param_dict["B"] = 11.0 + math.log(1.5)
    free_param_dict["omega_lim"] = 2.5

    return free_param_dict


def update_depend_param(param_dict: OrderedDict[str, float]) -> None:
    """
    Add/Update values of dependent parameters based on free parameters.

    Parameters
    ----------
    param_dict : Dict[str, float64]
        Dictionary that maps free & dependent parameters to their values.
    """

    f_min, k_me, Pdem = itemgetter("f_min", "k_me", "Pdem")(param_dict)

    # calculate dependent paramters
    param_dict["k_me01"] = 9 * k_me
    param_dict["k_me12"] = 6 * k_me
    param_dict["k_me23"] = k_me
    param_dict["gamma_me01"] = param_dict["k_me01"] / 20
    param_dict["gamma_me12"] = param_dict["k_me12"] / 20
    param_dict["gamma_me23"] = param_dict["k_me23"] / 20
    param_dict["gamma_dem"] = f_min * Pdem


# dynamic parameter
class DynamicParameters(object):
    """Manager of model dynamic parameters."""

    __free_param_dict = load_default_free_param_dict()  # store recommanded (default) values

    def __init__(self, **kwargs: float) -> None:
        """
        Constructor of class `DynamicParameters`.

        Parameters
        ----------
        **kwargs : float64
            Keyword arguments that specify the customized free parameter. e.g.
                
                >>> param = DynamicParameters(mu=1.0, sigma=3.0, Pt=1/2)

        Attributes
        ----------
        DynamicParameter.param_dict : Dict[str, float64]
            Dictionary that maps model parameters to their values.
        """

        assert set(kwargs).issubset(
            set(self.__free_param_dict)
        ), f"Acceptable paramter names: {', '.join(self.__free_param_dict)}."

        self.param_dict = copy.deepcopy(self.__free_param_dict)
        self.param_dict.update(kwargs)  # update free paramters
        self.__update_depend_param()  # update dependent paramters

    def set_free_param(self, **kwargs: float) -> None:
        """
        Modify the parameters of an existed model.

        Parameters
        ----------
        **kwargs : float64
            Keyword arguments that specify the model free parameter. If nothing passed, reset
            all free parameters to their default values.
        """

        if kwargs:
            assert set(kwargs).issubset(
                set(self.__free_param_dict)
            ), f"Acceptable paramter names: {', '.join(self.__free_param_dict)}."
            self.param_dict.update(kwargs)  # update free paramters
        else:
            self.param_dict.update(self.__free_param_dict)  # update free paramters

        self.__update_depend_param()  # update dependent paramters

    def get_param(self, param_names: Union[str, Tuple[str]] = "all") -> Union[float, Tuple[float, ...]]:
        """
        Get values of model parameters from built-in ordered dictionary.

        Parameters
        ----------
        param_names : str | Sequence[str] (default="all")
            Speficy the paramters that to be extract. If none, return all values as the same order of dictionary.
            Note that for string input, besides of single parameter name, here are some special subset names to
            be acceptable as well:
                "all" - all parameters (default) ;
                "free" - free parameters ;
                "depend" - dependent parameters ;
                "alphaT" - parameters to be used in cycle-dependent activation level computation ;
                "prot_fp" - parameters to be used in protein fixed points computation.

        Returns
        -------
        param_vals : float64 | Tuple[float64, ...]
            Values of queried parameters.
        """

        if isinstance(param_names, (List, Tuple)):
            param_vals = itemgetter(*param_names)(self.param_dict)
        else:
            if param_names == "all":
                param_vals = tuple(self.param_dict.values())
            elif param_names == "free":
                param_vals = tuple(self.param_dict.values())[:21]
            elif param_names == "depend":
                param_vals = tuple(self.param_dict.values())[21:]
            elif param_names == "alphaT":
                param_vals = itemgetter("cell_cycle", "cc0", "mu", "alpha_lim")(self.param_dict)
            elif param_names == "prot_fp":
                param_vals = itemgetter(
                    "cell_cycle",
                    "cc0",
                    "mu",
                    "alpha_lim",
                    "epsilon",
                    "sigma",
                    "Kd",
                    "f_min",
                    "f_max",
                    "f_lim",
                    "Pt",
                    "gamma_transcr",
                    "prot_per_transcr",
                    "kappa",
                )(self.param_dict)
            else:
                param_vals = self.param_dict[param_names]

        return param_vals

    def __update_depend_param(self) -> None:
        return update_depend_param(param_dict=self.param_dict)

    @classmethod
    def default_free_param_dict(cls) -> OrderedDict[str, float]:
        """Return a dictionary that store the recommanded value of model free parameters."""
        return cls.__free_param_dict
