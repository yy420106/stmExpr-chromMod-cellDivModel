function [geneExpr_records, meState_records, transcrT_records, alpha_end] = gillespie_ssa( ...
    geneExpr0, meState0, time_to_next_repl0, dyp, time_records, act_on_gene, buffer, alpha0, ...
    sizeFactor, calc_alpha_end)
    % Description:
    %   Gillespie stochastic simulation algorithm (SSA).
    %
    % Parameters:
    %   geneExpr0: float
    %       Initial gene expression quantity at the start of simulation.
    %   meState0: int array, size [1, H]
    %       Initial methylation state of each H3 histone at the start of simulation.
    %   time_to_next_repl0: float
    %       Time interval (unit: hour) between simulation initiation and the 1st DNA replication after that.
    %       Note that if it set to 0.0 (in most case), then 'geneExpr0' and 'meState0' refers to the model state
    %       at the very end of last cell cycle, and a cell division will do immediately.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    %   time_records: float array, size [1, T]
    %       Array of time points (unit: hour) at which to monitor the model state.
    %   act_on_gene: logical
    %       Whether to apply the master model (directly act on gene transcription) or alternative model (indirectly 
    %       act on gene transcription by epigentic modification).
    %   buffer: logical
    %     Determine whether to use buffering strategy for cycle-dependent activation level 'alpha'.
    %   alpha0: float
    %       Reference value for buffering 'alpha'.
    %   sizeFactor: float (default=1.1)
    %       Factor that controls the initial array size of return 'transcrT_records'. See 'Hints'.
    %   calc_alpha_end: logical (default=True)
    %       Determine whether to compute 'alpha' value at the end of simulation. This parameter is recommanded to
    %       set to False only if you want to avoid redundant work in parallel computing (function 
    %       'gillespie_ssa_parallel'), otherwise it should be always set to True by default.
    %
    % Returns:
    %   geneExpr_records: float array, size [1, T]
    %       1-D array, with entry t is the gene expression quantity at time_records[t]
    %   meState_records: int array, size [T, H]
    %       2-D array, with entry (t, h) is the methylation state of histone with index h at time_records[t]
    %   transcrT_records: float array, size [1, #]
    %       1-D array with enough length that store the time of every transcription event, empty spaces are
    %       filled with NaN. '#' indicates the length is nontrivial, with the computing formula shows below:
    %           length(transcrT_records) = sizeFactor * (time_records(end) - time_records(1)) * 60 * 60 * f_lim
    %   alpha_end: float
    %       When buffer strategy is used, return the value of 'alpha' at the end of simulation. This is useful when
    %       simulation stops in buffering stage before 'alpha' reach its target value.
    %
    % Hints:
    %   In this function, the size of array 'transcrT_records' (adjusted by 'sizeFactor') is pre-allocate, because
    %   it is always better than to dynamically adjust it every time. To ensure that the array 'transcrT_records'
    %   array is long enough to store all transcription time, 'sizeFactor' should not be too small. On the other hand,
    %   if 'sizeFactor' is too big, initialize a long array also cause time waste.
    
    arguments
        geneExpr0 double
        meState0 (1,:) double
        time_to_next_repl0 double
        dyp struct
        time_records (1,:) double
        act_on_gene logical
        buffer logical
        alpha0 double
        sizeFactor double = 1.1
        calc_alpha_end logical = true
    end
    
    % initialize output
    geneExpr_records = zeros(1, length(time_records));
    geneExpr_records(1) = geneExpr0;
    
    meState_records = zeros(length(time_records), length(meState0));
    meState_records(1, :) = meState0;
    
    transcrT_records = nan(1, round(sizeFactor * (time_records(end) - time_records(1)) * 60 * 60 * dyp.f_lim));  % huge enough
    transcr_idx = 1;

    % temporary variables
    curr_geneExpr = geneExpr0; % current gene expression
    curr_meState = meState0; % current methylation state
    curr_time = time_records(1); % current time [hour]
    next_time_records_idx = 2; % next time index at which gene expression and methylation state need to be recorded
    next_repl_time = time_records(1) + time_to_next_repl0; % next DNA replication time [hour]

    % evolution loop
    while next_time_records_idx <= length(time_records)
        while curr_time < time_records(next_time_records_idx)
            % draw the event and interval time
            [deltaT, event_class, histone_idx] = gillespie_draw( ...
                curr_geneExpr, curr_meState, dyp, act_on_gene, buffer, [alpha0, curr_time - time_records(1)]);
            delta_time = deltaT / (60 * 60); % convert unit from second to hour

            % save current model state before evolution
            prev_geneExpr = curr_geneExpr;
            prev_meState = curr_meState;

            if curr_time + delta_time < next_repl_time
                curr_time = curr_time + delta_time; % update time

                % update current methylation state
                if event_class == 1
                    curr_meState(histone_idx) = curr_meState(histone_idx) + 1; % H3 methylation
                elseif event_class == 2
                    curr_meState(histone_idx) = curr_meState(histone_idx) - 1; % H3 demethylation
                elseif event_class == 3
                    curr_geneExpr = curr_geneExpr + dyp.prot_per_transcr; % gene expression
                    
                    % transcription-coupled demethylation
                    hist_idx = (rand(size(curr_meState)) < dyp.Pdem) & (curr_meState > 0);
                    curr_meState(hist_idx) = curr_meState(hist_idx) - 1;

                    % transcription-coupled histone exchange
                    nuc_idx = find(rand(1, length(curr_meState) / 2) > (1 - dyp.Pex) ^ 2);
                    curr_meState(2 * nuc_idx - 1) = 0;
                    curr_meState(2 * nuc_idx) = 0;

                    % record transcription time
                    transcrT_records(transcr_idx) = curr_time;
                    transcr_idx = transcr_idx + 1;
                else
                    curr_geneExpr = max(curr_geneExpr - 1, 0); % protein degradation (filter negative value)
                end
            else
                % reset time when meeting cell cycle
                curr_time = next_repl_time;

                % nucleosomes reassemble after DNA-replication
                nuc_idx = find(rand(1, length(curr_meState) / 2) > 0.5);
                curr_meState(2 * nuc_idx - 1) = 0;
                curr_meState(2 * nuc_idx) = 0;

                % update next replication time
                next_repl_time = next_repl_time + dyp.cell_cycle;
            end
        end

        % update methylation state from next recording time to current time
        temp_idx = find(time_records > curr_time, 1); % the first time index after current time
        if isempty(temp_idx)
            temp_idx = length(time_records) + 1;
        end
        geneExpr_records(next_time_records_idx:temp_idx - 1) = prev_geneExpr;
        meState_records(next_time_records_idx:temp_idx - 1, :) = repmat(prev_meState, temp_idx - next_time_records_idx, 1);

        % update next recording time index
        next_time_records_idx = temp_idx;
    end

    % compute alpha at the end of simulation
    alpha = max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim);  % stable cycle-dependent activation level
    if buffer && calc_alpha_end
        if alpha0 <= alpha
            alpha_end = min(alpha0 * exp(dyp.alpha_expk * (time_records(end) - time_records(1))), alpha);
        else
            alpha_end = max(alpha0 * exp(-dyp.alpha_expk * (time_records(end) - time_records(1))), alpha);
        end
    else
        alpha_end = alpha;
    end
end
