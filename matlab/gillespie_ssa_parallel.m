function [samples_geneExpr, samples_meState, samples_transcrT, alpha_end] = gillespie_ssa_parallel( ...
    geneExpr0, meState0, time_to_next_repl0, dyp, time_records, act_on_gene, buffer, alpha0, sizeFactor)
    % Description:
    %   Parallel version of Gillespie stochastic simulation algorithm (SSA), each sample will form an independent
    %   parallel computing branch.
    %
    % Parameters  (See function 'gillespie_ssa' for more details):
    %   geneExpr0: float array, size [1, N]
    %       Initial gene expression quantity of each sample at the start of simulation
    %   meState0: int array, size [N, H]
    %       Initial methylation state of each H3 histone in each sample at the start of simulation.
    %   time_to_next_repl0: float array, size [1, N]
    %       Time interval (unit: hour) between simulation initiation and the 1st DNA replication after that
    %       of each sample.
    %   dyp: struct
    %   time_records: float array, size [1, T]
    %   act_on_gene: logical
    %   buffer: logical
    %   alpha0: double
    %   sizeFactor: float (default=1.1)
    %
    % Returns (See function 'gillespie_ssa' for more details):
    %   samples_geneExpr: float array, size [N, T]
    %       2-D array, with entry (n, t) is the gene expression quantity at time_records[t] in trial[n].
    %   samples_meState: int array, size [N, T, H]
    %       3-D array, with entry (n, t, h) is the methylation state of histone with index h at time_records[t]
    %       in trial[n].
    %   samples_transcrT: float array, size [N, #]
    %       2-D array with enough length that store the time of every transcription event in each trial, empty
    %       spaces are filled with NaN. '#' indicates the length is nontrivial, which comuted as:
    %           length(transcrT_records) = sizeFactor * (time_records(end) - time_records(1)) * 60 * 60 * f_lim
    %   alpha_end: float
    
    arguments
        geneExpr0 (1,:) double
        meState0 (:,:) double
        time_to_next_repl0 (1,:) double
        dyp struct
        time_records (1,:) double
        act_on_gene logical
        buffer logical
        alpha0 double
        sizeFactor double = 1.1
    end

    % check
    assert(length(geneExpr0) == size(meState0, 1) && size(meState0, 1) == length(time_to_next_repl0));

    N = size(meState0, 1);
    H = size(meState0, 2);

    % initialize output
    samples_geneExpr = zeros(N, length(time_records));
    samples_meState = zeros(N, length(time_records), H);
    samples_transcrT = zeros(N, round(sizeFactor * (time_records(end) - time_records(1)) * 60 * 60 * dyp.f_lim));

    % parallel loop
    %updateWaitbar = waitbarParfor(N, "Processing ...");
    parfor n = 1:N
        [samples_geneExpr(n,:), samples_meState(n,:,:), samples_transcrT(n,:), ~] = gillespie_ssa(geneExpr0(n), ...
            meState0(n,:), time_to_next_repl0(n), dyp, time_records, act_on_gene, buffer, alpha0, sizeFactor, false);
        %updateWaitbar();
    end

    % compute alpha at the end of simulation
    alpha = max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim);  % stable cycle-dependent activation level
    if buffer
        if alpha0 <= alpha
            alpha_end = min(alpha0 * exp(dyp.alpha_expk * (time_records(end) - time_records(1))), alpha);
        else
            alpha_end = max(alpha0 * exp(-dyp.alpha_expk * (time_records(end) - time_records(1))), alpha);
        end
    else
        alpha_end = alpha;
    end
end
