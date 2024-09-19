function [deltaT, event_class, histone_idx] = gillespie_draw( ...
    geneExpr, meState, dyp, act_on_gene, buffer, buffer_ref)
    % Description:
    %   Draws a event and the time it took to do that event in a poisson process.
    %
    % Parameters:
    %   geneExpr: float
    %       Gene expression quantity.
    %   meState: int array, size [1, H]
    %       Methylation state of chromatin H3 histone at target gene locus.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    %   act_on_gene: logical
    %       Whether to apply the master model (directly act on gene transcription) or alternative model 
    %       (indirectly act on gene transcription by epigentic modification).
    %   buffer: logical
    %       Determine whether to use buffering strategy for cycle-dependent activation level 'alpha'.
    %   buffer_ref: float array, size [1, 2]
    %       Reference information used for computing buffer value of 'alpha', in the order of [alpha0, time_delta],
    %       where 'alpha0' is reference value, 'time_delta' is time increment. This parameter is used only if 
    %       parameter 'buffer' is true.
    %
    % Returns:
    %   deltaT: float
    %       Time interval (unit: sec) for next comming event.
    %   event_class: int (1, 2, 3, 4)
    %       Next event marks, with 1, 2, 3, 4 represents H3 methylation, H3 demethylation, gene transcription
    %       and protein degradation, respectively.
    %   histone_idx: int (1, 2, ... , H-1, H)
    %       Histone index in which next event to be occured. Note that this value is useful only if methylation
    %       or demethylation occurs, otherwise it is set to 0 and meaningless.

    arguments
        geneExpr double
        meState (1,:) double
        dyp struct
        act_on_gene logical
        buffer logical
        buffer_ref (1,2) double
    end

    % compute propensity distribution
    [mePropensity, demPropensity, exprPropensity, pdgrPropensity] = gillespie_get_propensities( ...
        geneExpr, meState, dyp, act_on_gene, buffer, buffer_ref);
    props = [mePropensity, demPropensity, exprPropensity, pdgrPropensity];
    props_sum = sum(props);

    % compute next time
    deltaT = exprnd(1/props_sum);
    % draw event from this distribution
    q = rand * props_sum;
    p_sum = 0;
    idx = 0;
    while p_sum <= q
        p_sum = p_sum + props(idx + 1);
        idx = idx + 1;
    end
    
    % classify event
    H = numel(meState);
    if idx <= 2 * H
        event_class = floor((idx - 1) / H) + 1;
        histone_idx = mod(idx - 1, H) + 1;
    else
        event_class = 2 + idx - 2 * H;
        histone_idx = 0;
    end
end
