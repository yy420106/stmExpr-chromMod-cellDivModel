function plot_epigenetic_stability(N_SAMPLES, NUM_H3, dyp, TAG, MONITER_DAYS, save_dir)
    % Description:
    %   Plot multiple cell cycle to show model stability.
    %
    % Parameters
    %   N_SAMPLES: int
    %       Number of trials to be sampled.
    %   NUM_H3：int
    %       Histone number.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    %   TAG: string ("me0", "me3")
    %       Epigenetic modification state. Acceptable choices include "me0" and "me3".
    %   MONITER_DAYS: int
    %       Number of days to moniter model state.
    %   save_dir: string (default=pwd)
    %       Directory path for saving.

    arguments
        N_SAMPLES double
        NUM_H3 double
        dyp struct
        TAG string {mustBeMember(TAG, {'me0','me3'})}
        MONITER_DAYS double
        save_dir string = string(pwd)
    end
    
    % build model
    if strcmp(TAG, "me0")
        prot_critical_points = calc_prot_critical_points("max",dyp,4,true);
        geneExpr0 = round(prot_critical_points(end)) * ones(1, N_SAMPLES);
        meState0 = zeros(N_SAMPLES, NUM_H3);
    elseif strcmp(TAG, "me3")
        geneExpr0 = zeros(1, N_SAMPLES);
        meState0 = 3 * ones(N_SAMPLES, NUM_H3);
    else
        error("Error. \nInvalid input for 'TAG'.")
    end

    time_to_next_repl0 = zeros(1, N_SAMPLES);
    
    % Gillespie SSA
    alpha0 = max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim);
    time_records = linspace(0, MONITER_DAYS*24.0, MONITER_DAYS*24*60+1);
    [samples_geneExpr, samples_meState, samples_transcrT, ~] = gillespie_ssa_parallel( ...
        geneExpr0, meState0, time_to_next_repl0, dyp, time_records, true, true, alpha0);
    
    % plot
    f = figure("Name",strcat(TAG,"Stable"));
    tiledlayout(6,1,"TileSpacing","tight")
    for_color = {[1,0,0,0.3], [1,0,1,0.3], [0,1,1,0.3], [0,0,1,0.3]};
    for k=0:3
        nexttile
        samples_mek_ratio = sum(samples_meState==k,3) / NUM_H3;
        plot(time_records/24.0, samples_mek_ratio, "Color", for_color{k+1}, "LineWidth", 0.5)
        xlim([0, MONITER_DAYS])
        ylabel(['me', num2str(k)])
    end

    nexttile
    plot(time_records/24.0, samples_geneExpr, "Color", "green","LineWidth",0.5)
    xlim([0, MONITER_DAYS])
    ylabel("protein")
    
    nexttile
    [hist, bin_edges] = histcounts(samples_transcrT, 'BinEdges', linspace(0, MONITER_DAYS*24.0, MONITER_DAYS*24.0*4+1));
    samples_geneActivity = hist / size(samples_transcrT, 1);
    bin_mid = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
    bar(bin_mid/24.0, samples_geneActivity, 1, "black");
    xlim([0,MONITER_DAYS]);xlabel("Time (hour)")
    ylabel({"gene";"activity"})
    
    exportgraphics(f,fullfile(save_dir,strcat(TAG,"Stable.pdf")),...
        "ContentType","image","Resolution",800)
end