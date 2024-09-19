import math
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numba_progress as nbp
import numpy as np
from colorama import Fore, Style, init
from model import GeneChromModel
from plot_funcs1 import (
    plot_bistability_map,
    plot_cell_type_distribution,
    plot_evolution_heatmap,
    plot_schematic_diagram1,
    plot_schematic_diagram2,
    verbose_print,
)
from plot_funcs2 import plot_evolution_curves
from post_process_funcs import get_samples_bistability
from tqdm import tqdm
from util_funcs import calc_prot_fixed_points

__all__ = [
    "numba_progress_bar_set",
    "command_hint_print",
    "main_schmdg",
    "main_epistb",
    "main_dynmcyc",
    "main_divarrest",
    "main_rescue",
    "main_bimap",
]

# global variables & settings
init(autoreset=True)  # colorama mode: autoreset


def numba_progress_bar_set(total: int, disable: bool, leave: bool = True, n: int = 0) -> Dict[str, Any]:
    """Auxilary function. Build keywords argument dictionary for `numba_progress.ProgressBar`."""

    return dict(
        total=total,
        disable=disable,
        colour="white",
        desc=Fore.GREEN + Style.BRIGHT + "\tGillespie SSA" + (f"-{n}" if n else "") + Style.RESET_ALL,
        leave=leave,
        dynamic_ncols=False,
        ncols=128,
    )


def command_hint_print(cmd: str, quiet: bool, hint_type: str, **kwargs: Any) -> None:
    """
    Auxilary function. Print command hint according to different condition.

    Parameters
    ----------
    cmd : str
        Command name.
    quiet : bool
        Control flag.
    hint_type : str
        Type of hint information.
    **kwargs : Any
        Other necessary keywords argument. See detailed definition.
    """

    if hint_type == "init":
        verbose_print(message=f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} Initiating ...", quiet=quiet)
    
    elif hint_type == "end":
        time: Tuple[datetime, datetime] = kwargs["time"]
        verbose_print(
            message=(
                f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} All mission completed. "
                f"Program exits with total running time: {Fore.BLUE+Style.BRIGHT+str(time[1]-time[0])+Style.RESET_ALL}."
            ),
            quiet=quiet,
        )
    
    elif hint_type == "model":
        verbose_print(
            message=(
                f"{Fore.YELLOW + Style.BRIGHT}[{cmd}]{Style.RESET_ALL} "
                f"Start modeling and simulation by {Fore.GREEN+Style.BRIGHT}Gillespie SSA{Style.RESET_ALL} ..."
            ),
            quiet=quiet,
        )
    
    elif hint_type == "data":
        data_name: str = kwargs["data_name"]
        verbose_print(
            message=(
                f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} "
                f"Save raw data {Fore.MAGENTA+Style.BRIGHT+data_name+Style.RESET_ALL} to target directory successfully."
            ),
            quiet=quiet,
        )
    
    elif hint_type == "plot":
        func_name: str = kwargs["func_name"]
        extra: bool = kwargs["extra"]
        verbose_print(
            message=(
                f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} "
                f"Plot {"additional " if extra else ""}figure using function {Fore.GREEN+Style.BRIGHT+func_name+Style.RESET_ALL} ..."
            ),
            quiet=quiet,
        )
    
    elif hint_type == "export":
        fig_name: str = kwargs["fig_name"]
        extra: bool = kwargs["extra"]
        verbose_print(
            message=(
                f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} "
                f"Export {"additional " if extra else ""}figure {Fore.MAGENTA+Style.BRIGHT+fig_name+Style.RESET_ALL} to target directory successfully."
            ),
            quiet=quiet,
        )
    
    elif hint_type == "skip":
        verbose_print(
            message=f"{Fore.YELLOW+Style.BRIGHT}[{cmd}]{Style.RESET_ALL} Skip additional analysis and plot.",
            quiet=quiet,
        )
    
    else:
        raise ValueError("Unrecognizable hint type.")


def main_schmdg(
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `schmdg`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()
    command_hint_print("schmdg", quiet, "init")

    name1 = "schematicDiagram1" if prefix is None else prefix + "1"
    command_hint_print("schmdg", quiet, "plot", func_name="plot_schematic_diagram1", extra=False)
    plot_schematic_diagram1(name1, size, format, dpi, save_dir, quiet)
    command_hint_print("schmdg", quiet, "export", fig_name=name1, extra=False)

    name2 = "schematicDiagram2" if prefix is None else prefix + "2"
    command_hint_print("schmdg", quiet, "plot", func_name="plot_schematic_diagram2", extra=False)
    plot_schematic_diagram2(name2, size, format, dpi, save_dir, quiet)
    command_hint_print("schmdg", quiet, "export", fig_name=name2, extra=False)

    m_end = datetime.now()
    command_hint_print("schmdg", quiet, "end", time=(m_start, m_end))


def main_epistb(
    num_samples: int,
    epi_tag: List[Literal[0, 1, 2, 3]],
    monitor_days: float,
    alternative: bool,
    update_interval: float,
    concise: bool,
    timeid_step: Optional[int],
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    prefix2: Optional[str],
    size2: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `epistb`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()
    command_hint_print("epistb", quiet, "init")

    stem_pfp = calc_prot_fixed_points(0, *GeneChromModel().prot_fp_param, not alternative)[-1]

    for tag in epi_tag:
        if prefix is None:
            name = f"epiStable_me{tag}_alt" if alternative else f"epiStable_me{tag}"
        else:
            name = prefix + f"_me{tag}_alt" if alternative else prefix + f"_me{tag}"
        dname = f"rawdata_me{tag}_alt" if alternative else f"rawdata_me{tag}"

        command_hint_print("epistb", quiet, "model")
        model = GeneChromModel(
            N=num_samples,
            geneExpr=round(stem_pfp) if tag <= 1 else 0,
            meState_fastBuild=tag,
            time_to_next_repl=0,
            act_on_gene=not alternative,
        )

        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet)) as p_bar:
            time_records, samples_geneExpr, samples_meState, samples_transcrT, _ = model.evolve(
                ev_time=monitor_days * 24.0, time_step=update_interval, p_bar=p_bar
            )

        np.savez_compressed(
            os.path.join(save_dir, f"{dname}.npz"), time_records, samples_geneExpr, samples_meState, samples_transcrT
        )
        command_hint_print("epistb", quiet, "data", data_name=dname)

        command_hint_print("epistb", quiet, "plot", func_name="plot_evolution_curvese", extra=False)
        plot_evolution_curves(
            time_records=time_records,
            samples_geneExpr=samples_geneExpr,
            samples_meState=samples_meState,
            samples_transcrT=samples_transcrT,
            custom_mod_show={"me1": False, "me2": False, "gene": False},
            m_plot=True,
            geneExpr_min_zero=False,
            figName=name,
            figSize=size,
            figFormat=format,
            dpi=dpi,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("epistb", quiet, "export", fig_name=name, extra=False)

        if not concise:
            if prefix2 is None:
                name2 = f"evoHeatmap_me{tag}_alt" if alternative else f"evoHeatmap_me{tag}"
            else:
                name2 = prefix2 + f"_me{tag}_alt" if alternative else prefix2 + f"_me{tag}"

            command_hint_print("epistb", quiet, "plot", func_name="plot_evolution_heatmap", extra=True)
            plot_evolution_heatmap(samples_meState, timeid_step, name2, size2, format, dpi, save_dir, quiet)
            command_hint_print("epistb", quiet, "export", fig_name=name2, extra=True)
        else:
            command_hint_print("epistb", quiet, "skip")

    m_end = datetime.now()
    command_hint_print("epistb", quiet, "end", time=(m_start, m_end))


def main_dynmcyc(
    num_samples: int,
    dynamic_factor: List[float],
    equi_cycles: int,
    monitor_days: float,
    alternative: bool,
    update_interval: float,
    hist_interval: float,
    fluc_ratio: float,
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `dynmcyc`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()
    command_hint_print("dynmcyc", quiet, "init")

    stem_pfp = calc_prot_fixed_points(0, *GeneChromModel().prot_fp_param, not alternative)[-1]
    cell_cycle0 = GeneChromModel.default_free_param_dict().get("cell_cycle")

    for factor in dynamic_factor:
        annot = "accelerated" if factor < 1 else "decelerated"
        if prefix is None:
            name = f"cellDivision_{annot}_alt" if alternative else f"cellDivision_{annot}"
        else:
            name = prefix + f"_{annot}_alt" if alternative else prefix + f"_{annot}"
        dname = f"rawdata_{annot}_alt" if alternative else f"rawdata_{annot}"

        command_hint_print("dynmcyc", quiet, "model")
        model = GeneChromModel(
            N=num_samples,
            geneExpr=np.random.randint(
                low=math.ceil((1 - fluc_ratio) * stem_pfp),
                high=math.floor((1 + fluc_ratio) * stem_pfp),
                size=num_samples,
            ),
            meState_fastBuild=0,
            time_to_next_repl=0,
            time=-equi_cycles * cell_cycle0,
            act_on_gene=not alternative,
        )
        tc1 = model.time  # time-cut

        # pre-equilibration
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 1)) as p_bar:
            time_records, samples_geneExpr, samples_meState, samples_transcrT, alpha_end = model.evolve(
                ev_time=equi_cycles * cell_cycle0, time_step=update_interval, p_bar=p_bar
            )
        tc2 = model.time  # time-cut

        # accelerate/decelerate
        model.set_free_param(cell_cycle=factor * cell_cycle0)
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 2)) as p_bar:
            time_records_, samples_geneExpr_, samples_meState_, samples_transcrT_, _ = model.evolve(
                ev_time=monitor_days * 24.0, time_step=update_interval, alpha0=alpha_end, p_bar=p_bar
            )
        tc3 = model.time  # time-cut
        time_records = np.concatenate((time_records, time_records_[1:]))
        samples_geneExpr = np.concatenate((samples_geneExpr, samples_geneExpr_[:, 1:]), axis=1)
        samples_meState = np.concatenate((samples_meState, samples_meState_[:, 1:, :]), axis=1)
        samples_transcrT = np.concatenate((samples_transcrT, samples_transcrT_), axis=1)

        np.savez_compressed(
            os.path.join(save_dir, f"{dname}.npz"), time_records, samples_geneExpr, samples_meState, samples_transcrT
        )
        command_hint_print("dynmcyc", quiet, "data", data_name=dname)

        command_hint_print("dynmcyc", quiet, "plot", func_name="plot_evolution_curvese", extra=False)
        plot_evolution_curves(
            time_records=time_records,
            samples_geneExpr=samples_geneExpr,
            samples_meState=samples_meState,
            samples_transcrT=samples_transcrT,
            transcr_stat_bins=int((tc3 - tc1) / hist_interval),
            custom_mod_show={"me1": False, "me2": False},
            m_plot=True,
            bg_fill=True,
            title_annot_timecut=np.array([tc1, tc2, tc3]),
            title_annot_label=["normal", annot],
            geneExpr_min_zero=False,
            figName=name,
            figSize=size,
            figFormat=format,
            dpi=dpi,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("dynmcyc", quiet, "export", fig_name=name, extra=False)

    m_end = datetime.now()
    command_hint_print("dynmcyc", quiet, "end", time=(m_start, m_end))


def main_divarrest(
    num_samples: int,
    equi_cycles: int,
    monitor_days: float,
    arrest_days: List[float],
    plot_step: int,
    alternative: bool,
    update_interval: float,
    hist_interval: float,
    fluc_ratio: float,
    concise: bool,
    count_days: Optional[List[float]],
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    prefix2: Optional[str],
    size2: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `divarrest`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()  # main start
    command_hint_print("divarrest", quiet, "init")

    stem_pfp = calc_prot_fixed_points(0, *GeneChromModel().prot_fp_param, not alternative)[-1]
    cell_cycle0 = GeneChromModel.default_free_param_dict().get("cell_cycle")
    if not concise:
        diff_ratio_after_recov = np.full((len(count_days), len(arrest_days)), np.nan, np.float64)
        stem_ratio_after_recov = np.full((len(count_days), len(arrest_days)), np.nan, np.float64)

    for arstd_idx, arstd in enumerate(arrest_days):
        if prefix is None:
            name = f"divArrest_{arstd}Days_alt" if alternative else f"divArrest_{arstd}Days"
        else:
            name = prefix + f"_{arstd}Days_alt" if alternative else prefix + f"_{arstd}Days"
        dname = f"rawdata_{arstd}Days_alt" if alternative else f"rawdata_{arstd}Days"

        recvd = monitor_days - arstd  # number of recovery days

        command_hint_print("divarrest", quiet, "model")
        model = GeneChromModel(
            N=num_samples,
            geneExpr=np.random.randint(
                low=math.ceil((1 - fluc_ratio) * stem_pfp),
                high=math.floor((1 + fluc_ratio) * stem_pfp),
                size=num_samples,
            ),
            meState_fastBuild=0,
            time_to_next_repl=0,
            time=-equi_cycles * cell_cycle0,  # set negtive time for pre-equilibrium stage
            act_on_gene=not alternative,
        )
        tc1 = model.time  # time-cut

        # pre-equilibration
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 1)) as p_bar:
            time_records, samples_geneExpr, samples_meState, samples_transcrT, alpha_end = model.evolve(
                ev_time=equi_cycles * cell_cycle0, time_step=update_interval, p_bar=p_bar
            )
        tc2 = model.time  # time-cut

        # division arrest
        model.set_free_param(cell_cycle=np.inf)
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 2)) as p_bar:
            time_records_, samples_geneExpr_, samples_meState_, samples_transcrT_, alpha_end = model.evolve(
                ev_time=arstd * 24.0, time_step=update_interval, alpha0=alpha_end, p_bar=p_bar
            )
        tc3 = model.time  # time-cut
        time_records = np.concatenate((time_records, time_records_[1:]))
        samples_geneExpr = np.concatenate((samples_geneExpr, samples_geneExpr_[:, 1:]), axis=1)
        samples_meState = np.concatenate((samples_meState, samples_meState_[:, 1:, :]), axis=1)
        samples_transcrT = np.concatenate((samples_transcrT, samples_transcrT_), axis=1)

        # recovery
        model.set_free_param(cell_cycle=cell_cycle0)
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 3)) as p_bar:
            time_records_, samples_geneExpr_, samples_meState_, samples_transcrT_, _ = model.evolve(
                ev_time=recvd * 24.0, time_step=update_interval, alpha0=alpha_end, p_bar=p_bar
            )
        tc4 = model.time  # time-cut
        time_records = np.concatenate((time_records, time_records_[1:]))
        samples_geneExpr = np.concatenate((samples_geneExpr, samples_geneExpr_[:, 1:]), axis=1)
        samples_meState = np.concatenate((samples_meState, samples_meState_[:, 1:, :]), axis=1)
        samples_transcrT = np.concatenate((samples_transcrT, samples_transcrT_), axis=1)

        np.savez_compressed(
            os.path.join(save_dir, f"{dname}.npz"), time_records, samples_geneExpr, samples_meState, samples_transcrT
        )
        command_hint_print("divarrest", quiet, "data", data_name=dname)

        command_hint_print("divarrest", quiet, "plot", func_name="plot_evolution_curves", extra=False)
        plot_evolution_curves(
            time_records=time_records,
            samples_geneExpr=samples_geneExpr[::plot_step],
            samples_meState=samples_meState[::plot_step],
            samples_transcrT=samples_transcrT[::plot_step],
            transcr_stat_bins=round((tc4 - tc1) / hist_interval),
            custom_mod_show={"me1": False, "me2": False},
            m_plot=True,
            bg_fill=True,
            title_annot_timecut=[tc1, tc2, tc3, tc4],
            title_annot_label=["normal", "terminate", "recovery"],
            arrow_annot_pos=[0, stem_pfp],
            figName=name,
            figSize=size,
            dpi=dpi,
            figFormat=format,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("divarrest", quiet, "export", fig_name=name, extra=False)

        if not concise:
            # compute gene expression distribution
            for cnt_idx, cntd in enumerate(count_days):
                if cntd <= recvd:
                    tmp_idx = math.ceil(cntd / recvd * (time_records_.size - 1))
                    diff_ratio_after_recov[cnt_idx, arstd_idx] = sum(samples_geneExpr_[:, tmp_idx] == 0) / model.N
                    stem_ratio_after_recov[cnt_idx, arstd_idx] = (
                        sum(samples_geneExpr_[:, tmp_idx] >= 0.5 * stem_pfp) / model.N
                    )

    if not concise:
        if prefix2 is None:
            name2 = "cellTypeProportion_alt" if alternative else "cellTypeProportion"
        else:
            name2 = prefix2 + "_alt" if alternative else prefix2

        command_hint_print("divarrest", quiet, "plot", func_name="plot_cell_type_distribution", extra=True)
        plot_cell_type_distribution(
            arrest_days=arrest_days,
            count_days=count_days,
            diff_ratio=diff_ratio_after_recov,
            stem_ratio=stem_ratio_after_recov,
            figName=name2,
            figSize=size2,
            figFormat=format,
            dpi=dpi,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("divarrest", quiet, "export", fig_name=name2, extra=True)
    else:
        command_hint_print("divarrest", quiet, "skip")

    m_end = datetime.now()  # main end
    command_hint_print("divarrest", quiet, "end", time=(m_start, m_end))


def main_rescue(
    num_samples: int,
    rescue_strategy: List[Literal["M", "A", "S"]],
    arrest_days: int,
    treat_days: int,
    rest_days: int,
    prob_removal: float,
    affn_growth: float,
    prot_exogain: float,
    alternative: bool,
    update_interval: float,
    hist_interval: float,
    fluc_ratio: float,
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `rescue`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()  # main start
    command_hint_print("rescue", quiet, "init")

    stem_pfp = calc_prot_fixed_points(0, *GeneChromModel().prot_fp_param, not alternative)[-1]
    cell_cycle0 = GeneChromModel.default_free_param_dict().get("cell_cycle")

    for strategy in rescue_strategy:
        strategy_str = "removeMe" if strategy == "M" else ("addATH1" if strategy == "A" else "addSTM")
        strategy_label = "me3 removal" if strategy == "M" else ("add ATH1" if strategy == "A" else "add STM")
        if prefix is None:
            name = f"rescueBy_{strategy_str}_alt" if alternative else f"rescueBy_{strategy_str}"
        else:
            name = prefix + f"_{strategy_str}_alt" if alternative else prefix + f"_{strategy_str}"
        dname = f"rawdata_{strategy_str}_alt" if alternative else f"rawdata_{strategy_str}"

        command_hint_print("rescue", quiet, "model")
        model = GeneChromModel(
            N=num_samples,
            geneExpr=np.random.randint(
                low=math.ceil((1 - fluc_ratio) * stem_pfp),
                high=math.floor((1 + fluc_ratio) * stem_pfp),
                size=num_samples,
            ),
            meState_fastBuild=0,
            time_to_next_repl=0,
            time=-arrest_days * 24.0,
            act_on_gene=not alternative,
            cell_cycle=np.inf,
        )
        tc1 = model.time  # time-cut

        # stop division
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 1)) as p_bar:
            time_records, samples_geneExpr, samples_meState, samples_transcrT, alpha_end = model.evolve(
                ev_time=arrest_days * 24.0, time_step=update_interval, p_bar=p_bar
            )
        tc2 = model.time  # time-cut

        # recovery
        model.set_free_param(cell_cycle=cell_cycle0)
        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 2)) as p_bar:
            time_records_, samples_geneExpr_, samples_meState_, samples_transcrT_, alpha_end = model.evolve(
                ev_time=rest_days * 24.0, time_step=update_interval, alpha0=alpha_end, p_bar=p_bar
            )
        tc3 = model.time  # time-cut
        time_records = np.concatenate((time_records, time_records_[1:]))
        samples_geneExpr = np.concatenate((samples_geneExpr, samples_geneExpr_[:, 1:]), axis=1)
        samples_meState = np.concatenate((samples_meState, samples_meState_[:, 1:, :]), axis=1)
        samples_transcrT = np.concatenate((samples_transcrT, samples_transcrT_), axis=1)

        # rescue
        if strategy == "M":
            model.meState = np.where(np.random.rand(*model.meState.shape) < prob_removal, 0, model.meState)
        elif strategy == "A":
            model.set_free_param(epsilon=(1 + affn_growth) * model.param_dict["epsilon"])
        elif strategy == "S":
            model.geneExpr += prot_exogain * stem_pfp
        else:
            raise ValueError("unrecognizable rescue strategy is provided.")
        # update stable point because of parameters change
        stem_pfp_rescue = calc_prot_fixed_points(0, *model.prot_fp_param, act_on_gene=model.act_on_gene)[-1]

        with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, 3)) as p_bar:
            time_records_, samples_geneExpr_, samples_meState_, samples_transcrT_, _ = model.evolve(
                ev_time=treat_days * 24.0, time_step=update_interval, alpha0=alpha_end, p_bar=p_bar
            )
        tc4 = model.time  # time-cut
        time_records = np.concatenate((time_records, time_records_[1:]))
        samples_geneExpr = np.concatenate((samples_geneExpr, samples_geneExpr_[:, 1:]), axis=1)
        samples_meState = np.concatenate((samples_meState, samples_meState_[:, 1:, :]), axis=1)
        samples_transcrT = np.concatenate((samples_transcrT, samples_transcrT_), axis=1)

        np.savez_compressed(
            os.path.join(save_dir, f"{dname}.npz"), time_records, samples_geneExpr, samples_meState, samples_transcrT
        )
        command_hint_print("rescue", quiet, "data", data_name=dname)

        command_hint_print("rescue", quiet, "plot", func_name="plot_evolution_curves", extra=False)
        plot_evolution_curves(
            time_records=time_records,
            samples_geneExpr=samples_geneExpr,
            samples_meState=samples_meState,
            samples_transcrT=samples_transcrT,
            transcr_stat_bins=int((tc4 - tc1) / hist_interval),
            custom_mod_show={"me1": False, "me2": False},
            m_plot=True,
            bg_fill=True,
            title_annot_timecut=np.array([tc1, tc2, tc3, tc4]),
            title_annot_label=["terminate", "recovery", " + ".join(["recovery", strategy_label])],
            arrow_annot_pos=[0, stem_pfp_rescue],
            figName=name,
            figSize=size,
            figFormat=format,
            dpi=dpi,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("rescue", quiet, "export", fig_name=name, extra=False)

    m_end = datetime.now()
    command_hint_print("rescue", quiet, "end", time=(m_start, m_end))


def main_bimap(
    num_samples: int,
    cell_cycle: List[float],
    num_cycles: int,
    map_pixel: int,
    kme_logrange: List[float],
    pdem_logrange: List[float],
    alternative: bool,
    update_interval: float,
    quiet: bool,
    prefix: Optional[str],
    size: Optional[List[float]],
    format: Optional[List[str]],
    dpi: Optional[int],
    save_dir: str,
) -> None:
    """Implementation of main program subcommand `bimap`. See `main.py` for the parameter explanations."""

    m_start = datetime.now()  # main start
    command_hint_print("bimap", quiet, "init")

    # sample points in parameter space
    k_me_ticks = 10 ** np.linspace(*kme_logrange, map_pixel)
    Pdem_ticks = 10 ** np.linspace(*pdem_logrange, map_pixel)

    for cc in cell_cycle:
        if prefix is None:
            name = f"biMap_cellCycle{cc}h_alt" if alternative else f"biMap_cellCycle{cc}h"
        else:
            name = prefix + f"_cellCycle{cc}h_alt" if alternative else prefix + f"biMap_cellCycle{cc}h"
        dname = f"rawdata_cellCycle{cc}h_alt" if alternative else f"rawdata_cellCycle{cc}h"

        # initialize output
        bistabilityMap = np.zeros(shape=(map_pixel, map_pixel))

        command_hint_print("bimap", quiet, "model")
        stem_pfp = calc_prot_fixed_points(0, *GeneChromModel(cell_cycle=cc).prot_fp_param, not alternative)[-1]
        with tqdm(
            total=map_pixel**2,
            colour="white",
            desc=Fore.CYAN + Style.BRIGHT + "\tIterations" + Style.RESET_ALL,
            ncols=128,
            disable=quiet,
        ) as progress_bar:
            for i in range(map_pixel):
                for j in range(map_pixel):
                    model = GeneChromModel(
                        N=num_samples * 2,
                        geneExpr=np.array([round(stem_pfp)] * num_samples + [0] * num_samples),
                        meState_fastBuild=(0,) * num_samples + (3,) * num_samples,
                        time_to_next_repl=0,
                        act_on_gene=not alternative,
                        k_me=k_me_ticks[i],
                        Pdem=Pdem_ticks[j],
                        cell_cycle=cc,
                    )

                    with nbp.ProgressBar(**numba_progress_bar_set(model.N, quiet, False)) as p_bar:
                        time_records, _, samples_meState, _, _ = model.evolve(
                            ev_time=num_cycles * cc, time_step=update_interval, p_bar=p_bar
                        )
                    bistabilityMap[i, j] = get_samples_bistability(
                        samples_meState=samples_meState,
                        time_records=time_records,
                        time_to_first_repl=0.0,
                        cell_cycle=cc,
                        Pt=model.param_dict["Pt"],
                    )

                    progress_bar.update(1)

        np.savez_compressed(os.path.join(save_dir, f"{dname}.npz"), bistabilityMap)
        command_hint_print("bimap", quiet, "data", data_name=dname)

        command_hint_print("bimap", quiet, "plot", func_name="plot_bistability_map", extra=False)
        plot_bistability_map(
            bistabilityMap=bistabilityMap,
            axis0_ticks=k_me_ticks,
            axis1_ticks=Pdem_ticks,
            cell_cycle=cc,
            scatter_points=[GeneChromModel().get_param(param_names=("k_me", "Pdem"))],
            figName=name,
            figSize=size,
            figFormat=format,
            dpi=dpi,
            save_dir=save_dir,
            quiet=quiet,
        )
        command_hint_print("bimap", quiet, "export", fig_name=name, extra=False)

    m_end = datetime.now()
    command_hint_print("bimap", quiet, "end", time=(m_start, m_end))
