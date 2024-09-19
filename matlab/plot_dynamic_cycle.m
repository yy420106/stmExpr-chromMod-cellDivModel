function plot_dynamic_cycle(N_SAMPLES, NUM_H3, dyp, KF, EQUI_CYCLE, MONITER_DAYS, prot_fluc, save_dir)
    % Description:
    %   Plot model evolution curves after accelerate/decelerate cell division.
    %
    % Parameters:
    %   N_SAMPLES: int
    %       Number of trials to be sampled.
    %   NUM_H3：int
    %       Histone number.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    %   KF: float
    %       Multiclitive factor.
    %   EQUI_CYCLE: int
    %       Number of cell cycle for pre-equilibrium.
    %   MONITER_DAYS: int
    %       Number of days to moniter model state after pre-equilibrium.
    %   prot_fluc: float (default=0.5)
    %       Fluctuation percentage of protein from stable level when initializeing model.
    %   save_dir: string (default=pwd)
    %       Directory path for saving.

    arguments
        N_SAMPLES double
        NUM_H3 double
        dyp struct
        KF double
        EQUI_CYCLE double
        MONITER_DAYS double
        prot_fluc double = 0.5
        save_dir string = string(pwd)
    end
    
    prot_critical_points = calc_prot_critical_points("max",dyp,4,true);
    cp_max = prot_critical_points(end);

    % noraml
    t_start = -EQUI_CYCLE*dyp.cell_cycle;
    geneExpr0 = randi([ceil((1-prot_fluc)*cp_max),floor((1+prot_fluc)*cp_max)],1,N_SAMPLES);
    meState0 = zeros(N_SAMPLES, NUM_H3);
    time_to_next_repl0 = zeros(1, N_SAMPLES);
    time_records = linspace(t_start, 0, -t_start*60+1);
    [samples_geneExpr, samples_meState, samples_transcrT, alpha_end] = gillespie_ssa_parallel(geneExpr0, meState0, ...
        time_to_next_repl0, dyp, time_records, true, true, max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim));
    time_to_next_repl_end = calc_time_to_next_repl_after_ev(time_to_next_repl0,dyp.cell_cycle,-t_start);
    
    % change cell division
    t_end = MONITER_DAYS*24.0;
    geneExpr0 = samples_geneExpr(:,end)';
    meState0 = squeeze(samples_meState(:,end,:));
    time_to_next_repl0 = calc_time_to_next_repl_after_ccc(time_to_next_repl_end,dyp.cell_cycle,KF*dyp.cell_cycle);
    dyp = set_free_param(dyp,"cell_cycle",KF*dyp.cell_cycle);
    time_records_ = linspace(0, t_end, t_end*60+1);
    [samples_geneExpr_, samples_meState_, samples_transcrT_, ~] = gillespie_ssa_parallel( ...
        geneExpr0, meState0, time_to_next_repl0, dyp, time_records_, true, true, alpha_end);
    
    % concatenate
    time_records = cat(2,time_records,time_records_(2:end));
    samples_geneExpr = cat(2,samples_geneExpr, samples_geneExpr_(:,2:end));
    samples_meState = cat(2,samples_meState,samples_meState_(:,2:end,:));
    samples_transcrT = cat(2,samples_transcrT, samples_transcrT_);
    
    % plot
    if KF<1
        TAG="accelerated";
    else
        TAG="decelerated";
    end

    f = figure("Name",strcat(TAG,"Division"));
    tiledlayout(6,1,"TileSpacing","tight")
    for_color = {[1,0,0,0.3], [1,0,1,0.3], [0,1,1,0.3], [0,0,1,0.3]};
    for k=0:3
        nexttile
        xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
        hold on
        samples_mek_ratio = sum(samples_meState==k,3) / NUM_H3;
        plot(time_records/24.0, samples_mek_ratio, "Color", for_color{k+1}, "LineWidth", 0.5)
        xlim([t_start,t_end]/24)
        ylabel(['me', num2str(k)])
    end

    nexttile
    xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
    hold on
    plot(time_records/24.0, samples_geneExpr, "Color", "green", "LineWidth", 0.5)
    xlim([t_start,t_end]/24)
    ylabel("protein")
    
    nexttile
    xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
    hold on
    [hist, bin_edges] = histcounts(samples_transcrT, 'BinEdges', linspace(t_start, t_end, (t_end-t_start)*4+1));
    samples_geneActivity = hist / size(samples_transcrT, 1);
    bin_mid = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
    bar(bin_mid/24.0, samples_geneActivity, 1, "black");
    xlim([t_start,t_end]/24);xlabel("Time (hour)")
    ylabel({"gene";"activity"})
    
    exportgraphics(f,fullfile(save_dir,strcat(TAG,"Division.pdf")),...
        "ContentType","image","Resolution",800)
end