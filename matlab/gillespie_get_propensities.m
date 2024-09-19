function [mePropensity, demPropensity, exprPropensity, pdgrPropensity] = gillespie_get_propensities( ...
    geneExpr, meState, dyp, act_on_gene, buffer, buffer_ref)
    % Description:
    %   Compute propensities of H3 methylation, H3 demethylation, gene transcription and protein degradation
    %   based on the current gene expression, chromatin methylation state and model parameters.
    %
    % Parameters (See function 'gillespie_draw' for more details):
    %   geneExpr: float
    %   meState: int array, size [1, H]
    %   dyp: struct
    %   act_on_gene: logical
    %   buffer: logical
    %   buffer_ref: float array, size [1, 2]
    %
    % Returns:
    %   mePropensity: float array, shape [1, H]
    %       Methylation propensity for each H3 histone.
    %   demPropensity: float array, shape [1, H]
    %       Demethylation propensity for each H3 histone.
    %   exprPropensity: float
    %       Gene transcription propensity.
    %   pdgrPropensity: float
    %       Protein degradation propensity.
    
    arguments
        geneExpr double
        meState (1,:) double
        dyp struct
        act_on_gene logical
        buffer logical
        buffer_ref (1,2) double
    end

    H = numel(meState);  % histone number

    Pme23 = sum(meState >= 2) / H;  % Percentage of repressive methylation marks (me2/me3)
    
    E = zeros(1, H); % Neighbor enhancement
    for idx = 1:H
        if mod(idx, 2) == 1
            neighbor_meState = [meState(max(idx - 2, 1):idx - 1), meState(idx + 1:min(idx + 3, H))];
        else
            neighbor_meState = [meState(max(idx - 3, 1):idx - 1), meState(idx + 1:min(idx + 2, H))];
        end
        E(idx) = dyp.rho * sum(neighbor_meState == 2) + sum(neighbor_meState == 3) + dyp.e_distal;
    end
    
    theta = dyp.epsilon * geneExpr ^ dyp.sigma / (dyp.Kd + geneExpr ^ dyp.sigma);  % Cofactor-dependent activation level

    if act_on_gene
        [alpha, omega] = deal(max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim), 1.0);
        if buffer
            alpha0 = buffer_ref(1);
            time_delta = buffer_ref(2);
            if alpha0 <= alpha
                alpha = min(alpha0 * exp(dyp.alpha_expk * time_delta), alpha);
            else
                alpha = max(alpha0 * exp(-dyp.alpha_expk * time_delta), alpha);
            end
        end
    else
        [alpha, omega] = deal(1.0, dyp.omega_lim / (1 + exp(-dyp.A * dyp.cell_cycle + dyp.B)));
    end

    % compute propensities
    mePropensity = omega * dyp.beta * ( ...
        (dyp.gamma_me01 + dyp.k_me01 * E) .* (meState == 0) + ...
        (dyp.gamma_me12 + dyp.k_me12 * E) .* (meState == 1) + ...
        (dyp.gamma_me23 + dyp.k_me23 * E) .* (meState == 2) ...
        );
    demPropensity = dyp.gamma_dem * (meState > 0);
    exprPropensity = min( ...
        alpha * theta * (dyp.f_max - min(Pme23 / dyp.Pt, 1) * (dyp.f_max - dyp.f_min)), dyp.f_lim ...
        ) + dyp.gamma_transcr;
    pdgrPropensity = geneExpr * dyp.kappa;
end
