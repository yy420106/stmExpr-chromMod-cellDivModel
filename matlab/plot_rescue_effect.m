function plot_rescue_effect(N_SAMPLES, NUM_H3, dyp, ARREST_DAYS, MONITER_DAYS, REST_DAYS, ...
    prot_fluc, Peff, eps_grate, save_dir)
    % Description:
    %   Plot model evolution curves after long time of division arrest using 2 different rescue strategies.
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
    %   ARREST_DAYS: int
    %       Number of days for cell division arrest.
    %   MONITER_DAYS: int
    %       Number of days to moniter model state after pre-equilibrium.
    %   REST_DAYS: int array
    %       Rest days before adding extra treatment.
    %   prot_fluc: float (default=0.5)
    %       Fluctuation percentage of protein from stable level when initializeing model.
    %   Peff: float64 (default=0.8)
    %       Efficiency of 3-methylation removal, defined as the probability.
    %   eps_grate: float64(default=0.3)
    %       Relative growth rate of model free parameter 'epsilon' because of ATH1 adding.
    %   save_dir: string (default=pwd)
    %       Directory path for saving.

    arguments
        N_SAMPLES double
        NUM_H3 double
        dyp struct
        ARREST_DAYS double
        MONITER_DAYS double
        REST_DAYS (1,:) double
        prot_fluc double = 0.5
        Peff double = 0.8
        eps_grate double = 0.3
        save_dir string = string(pwd)
    end
    
    assert(REST_DAYS <= MONITER_DAYS)
    prot_critical_points = calc_prot_critical_points("max",dyp,4,true);
    cp_max = prot_critical_points(end);
    cell_cycle0 = dyp.cell_cycle;

    % stop division
    t_start = -ARREST_DAYS * 24.0;
    geneExpr0 = randi([ceil((1-prot_fluc)*cp_max),floor((1+prot_fluc)*cp_max)],1,N_SAMPLES);
    meState0 = zeros(N_SAMPLES, NUM_H3);
    time_to_next_repl0 = Inf(1, N_SAMPLES);
    dyp = set_free_param(dyp,"cell_cycle",Inf);
    time_records = linspace(t_start, 0, -t_start*60+1);
    [samples_geneExpr, samples_meState, samples_transcrT, alpha_end] = gillespie_ssa_parallel(geneExpr0, meState0, ...
        time_to_next_repl0, dyp, time_records, true, true, max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim));
    time_to_next_repl_end = calc_time_to_next_repl_after_ev(time_to_next_repl0,dyp.cell_cycle,-t_start);

    % recovery & rest
    t_c2 = REST_DAYS*24.0;
    geneExpr0 = samples_geneExpr(:,end)';
    meState0 = squeeze(samples_meState(:,end,:));
    time_to_next_repl0 = calc_time_to_next_repl_after_ccc(time_to_next_repl_end,dyp.cell_cycle,cell_cycle0);
    dyp = set_free_param(dyp,"cell_cycle",cell_cycle0);
    time_records_ = linspace(0, t_c2, t_c2*60+1);
    [samples_geneExpr_, samples_meState_, samples_transcrT_, alpha_end] = gillespie_ssa_parallel( ...
        geneExpr0, meState0, time_to_next_repl0, dyp, time_records_, true, true, alpha_end);
    time_to_next_repl_end = calc_time_to_next_repl_after_ev(time_to_next_repl0,dyp.cell_cycle,t_c2);
        
    time_records = cat(2,time_records,time_records_(2:end));
    samples_geneExpr = cat(2,samples_geneExpr, samples_geneExpr_(:,2:end));
    samples_meState = cat(2,samples_meState,samples_meState_(:,2:end,:));
    samples_transcrT = cat(2,samples_transcrT, samples_transcrT_);

    % rescue
    for cond=["removeMe3", "addATH1"]
        t_end = MONITER_DAYS*24.0;
        geneExpr0_cond = samples_geneExpr_(:,end)';
        meState0_cond = squeeze(samples_meState_(:,end,:));
        time_to_next_repl0_cond = time_to_next_repl_end;
        dyp_cond = dyp;
        time_records_cond = linspace(t_c2, t_end, (t_end-t_c2)*60+1);
        if strcmp(cond, "removeMe3")
            rm_idx = find(rand(1,N_SAMPLES)<Peff);
            meState0_cond(rm_idx,:) = repmat(zeros(1,NUM_H3),length(rm_idx),1);
        elseif strcmp(cond, "addATH1")
            dyp_cond = set_free_param(dyp_cond, "epsilon",(1+eps_grate)*dyp_cond.epsilon);
        end
        [samples_geneExpr_cond, samples_meState_cond, samples_transcrT_cond, ~] = gillespie_ssa_parallel( ...
            geneExpr0_cond, meState0_cond, time_to_next_repl0_cond, dyp_cond, time_records_cond, true, true, alpha_end);
        
        time_records_total = cat(2,time_records,time_records_cond(2:end));
        samples_geneExpr_total = cat(2,samples_geneExpr, samples_geneExpr_cond(:,2:end));
        samples_meState_total = cat(2,samples_meState,samples_meState_cond(:,2:end,:));
        samples_transcrT_total = cat(2,samples_transcrT, samples_transcrT_cond);
        
        % plot evolution curve
        f=figure("Name", cond);
        tiledlayout(6,1,"TileSpacing","tight")
        for_color = {[1,0,0,0.3], [1,0,1,0.3], [0,1,1,0.3], [0,0,1,0.3]};
        for k=0:3
            nexttile
            xregion(0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
            hold on
            xline(t_c2/24.0,"--","Color","black","LineWidth",1)
            samples_mek_ratio = sum(samples_meState_total==k,3) / NUM_H3;
            plot(time_records_total/24.0, samples_mek_ratio,"Color",for_color{k+1},"LineWidth",0.5)
            xlim([t_start,t_end]/24)
            ylabel(['me', num2str(k)])
        end

        nexttile
        xregion(0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
        hold on
        xline(t_c2/24.0,"--","Color","black","LineWidth",1)
        plot(time_records_total/24.0, samples_geneExpr_total, "Color", "green","LineWidth",0.5)
        xlim([t_start,t_end]/24)
        ylabel("protein")
    
        nexttile
        xregion(0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
        hold on
        xline(t_c2/24.0,"--","Color","black","LineWidth",1)
        [hist, bin_edges] = histcounts(samples_transcrT_total,'BinEdges',linspace(t_start,t_end,(t_end-t_start)*4+1));
        samples_geneActivity = hist / size(samples_transcrT_total, 1);
        bin_mid = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
        bar(bin_mid/24.0, samples_geneActivity, 1, "black");
        xlim([t_start,t_end]/24);xlabel("Time (hour)")
        ylabel("gene activity")

        exportgraphics(f,fullfile(save_dir,strcat(cond, ".pdf")),...
            "ContentType","image","Resolution",800)
    end
end