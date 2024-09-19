function plot_division_arrest(N_SAMPLES, NUM_H3, dyp, EQUI_CYCLE, MONITER_DAYS, TEST_DAYS, ...
    CELL_STAT_DAYS_AFTER_ARREST, disp_intv, act_on_gene, prot_fluc, save_dir)
    % Description:
    %   Plot model evolution curves before/after different length of cell division arrest & the change of
    %   cell type distribution after that.
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
    %   EQUI_CYCLE: int
    %       Number of cell cycle for pre-equilibrium.
    %   MONITER_DAYS: int
    %       Number of days to moniter model state after pre-equilibrium.
    %   TEST_DAYS: int array
    %       Sampling days of Cell division arrest in plotting.
    %   CELL_STAT_DYAS_AFTER_ARREST: int array
    %       Time (days) to collect cell type proportion data after recovery of cell division.
    %   disp_intv: int (default=1)
    %       Interval of curves to be show in model evolution plot.
    %   act_on_gene: bool (default=True)
    %       Whether to apply the master model (directly act on gene transcription) or alternative model 
    %       (indirectly act on gene transcription by epigentic modification).
    %   prot_fluc: float (default=0.5)
    %       Fluctuation percentage of protein from stable level when initializeing model.
    %   save_dir: string (default=pwd)
    %       Directory path for saving.

    arguments
        N_SAMPLES double
        NUM_H3 double
        dyp struct
        EQUI_CYCLE double
        MONITER_DAYS double
        TEST_DAYS (1,:) double
        CELL_STAT_DAYS_AFTER_ARREST (1,:) double
        disp_intv double = 1
        act_on_gene logical = true
        prot_fluc double = 0.5
        save_dir string = string(pwd)
    end
    
    assert(max(TEST_DAYS)+max(CELL_STAT_DAYS_AFTER_ARREST) <= MONITER_DAYS)
    prot_critical_points = calc_prot_critical_points("max",dyp,4,true);
    cp_max = prot_critical_points(end);
    cell_cycle0 = dyp.cell_cycle;

    % initialize output
    [diff_ratio_after_recov, stem_ratio_after_recov] = deal(zeros(length(CELL_STAT_DAYS_AFTER_ARREST), length(TEST_DAYS)));

    for test_idx=1:length(TEST_DAYS)
        N_TERMIN_DAYS = TEST_DAYS(test_idx);
        
        % normal division
        t_start = -EQUI_CYCLE * dyp.cell_cycle;
        geneExpr0 = randi([ceil((1-prot_fluc)*cp_max),floor((1+prot_fluc)*cp_max)],1,N_SAMPLES);
        meState0 = zeros(N_SAMPLES, NUM_H3);
        time_to_next_repl0 = zeros(1, N_SAMPLES);
        time_records = linspace(t_start, 0, -t_start*60+1);
        [samples_geneExpr, samples_meState, samples_transcrT, alpha_end] = gillespie_ssa_parallel(geneExpr0, meState0, ...
            time_to_next_repl0, dyp, time_records, act_on_gene, true, max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim));
        time_to_next_repl_end = calc_time_to_next_repl_after_ev(time_to_next_repl0,dyp.cell_cycle,-t_start);

        % stop division
        t_c2 = N_TERMIN_DAYS*24.0;
        geneExpr0 = samples_geneExpr(:,end)';
        meState0 = squeeze(samples_meState(:,end,:));
        time_to_next_repl0 = calc_time_to_next_repl_after_ccc(time_to_next_repl_end,dyp.cell_cycle,Inf);
        dyp = set_free_param(dyp,"cell_cycle",Inf);
        time_records_ = linspace(0, t_c2, t_c2*60+1);
        [samples_geneExpr_, samples_meState_, samples_transcrT_, alpha_end] = gillespie_ssa_parallel( ...
            geneExpr0, meState0, time_to_next_repl0, dyp, time_records_, act_on_gene, true, alpha_end);
        time_to_next_repl_end = calc_time_to_next_repl_after_ev(time_to_next_repl0,dyp.cell_cycle,t_c2);
        
        time_records = cat(2,time_records,time_records_(2:end));
        samples_geneExpr = cat(2,samples_geneExpr, samples_geneExpr_(:,2:end));
        samples_meState = cat(2,samples_meState,samples_meState_(:,2:end,:));
        samples_transcrT = cat(2,samples_transcrT, samples_transcrT_);

        % recovery division
        t_end = MONITER_DAYS*24.0;
        geneExpr0 = samples_geneExpr_(:,end)';
        meState0 = squeeze(samples_meState_(:,end,:));
        time_to_next_repl0 = calc_time_to_next_repl_after_ccc(time_to_next_repl_end,dyp.cell_cycle,cell_cycle0);
        dyp = set_free_param(dyp,"cell_cycle",cell_cycle0);
        time_records_ = linspace(t_c2, t_end, (t_end-t_c2)*60+1);
        [samples_geneExpr_, samples_meState_, samples_transcrT_, ~] = gillespie_ssa_parallel( ...
            geneExpr0, meState0, time_to_next_repl0, dyp, time_records_, act_on_gene, true, alpha_end);

        time_records = cat(2,time_records,time_records_(2:end));
        samples_geneExpr = cat(2,samples_geneExpr, samples_geneExpr_(:,2:end));
        samples_meState = cat(2,samples_meState,samples_meState_(:,2:end,:));
        samples_transcrT = cat(2,samples_transcrT, samples_transcrT_);

        for stat_idx=1:length(CELL_STAT_DAYS_AFTER_ARREST)
            time_idx = CELL_STAT_DAYS_AFTER_ARREST(stat_idx)*24.0*60+1;
            diff_ratio_after_recov(stat_idx, test_idx) = nnz(samples_geneExpr_(:, time_idx) == 0) / N_SAMPLES;
            stem_ratio_after_recov(stat_idx, test_idx) = nnz(samples_geneExpr_(:,time_idx) >= 0.5 * cp_max) / N_SAMPLES;
        end
        
        % plot evolution curve
        f = figure("Name",strcat(num2str(act_on_gene),"_teminate",num2str(N_TERMIN_DAYS),"Days"));
        tiledlayout(6,1,"TileSpacing","tight")
        for_color = {[1,0,0,0.3], [1,0,1,0.3], [0,1,1,0.3], [0,0,1,0.3]};
        for k=0:3
            nexttile
            xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
            hold on
            xregion(t_c2/24.0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
            samples_mek_ratio = sum(samples_meState==k,3) / NUM_H3;
            plot(time_records/24.0, samples_mek_ratio(1:disp_intv:end,:),"Color",for_color{k+1},"LineWidth",0.5)
            xlim([t_start,t_end]/24)
            ylabel(['me', num2str(k)])
        end

        nexttile
        xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
        hold on
        xregion(t_c2/24.0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
        plot(time_records/24.0, samples_geneExpr(1:disp_intv:end,:), "Color", "green","LineWidth",0.5)
        xlim([t_start,t_end]/24)
        ylabel("protein")
    
        nexttile
        xregion(t_start/24.0, 0,"FaceColor","yellow","FaceAlpha",0.2)
        hold on
        xregion(t_c2/24.0,t_end/24.0,"FaceColor","yellow","FaceAlpha",0.2)
        [hist, bin_edges] = histcounts(samples_transcrT(1:disp_intv:end,:), ...
            'BinEdges',linspace(t_start,t_end,(t_end-t_start)*4+1));
        samples_geneActivity = hist / size(samples_transcrT, 1);
        bin_mid = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
        bar(bin_mid/24.0, samples_geneActivity, 1, "black");
        xlim([t_start,t_end]/24);xlabel("Time (hour)")
        ylabel({"gene";"activity"})

        exportgraphics(f,fullfile(save_dir,strcat(num2str(act_on_gene),"_teminate",num2str(N_TERMIN_DAYS),"Days.pdf")),...
            "ContentType","vector","Resolution",800)
    end

    % plot cell type proportion
    h = figure("Name","cellTypeProportion");
    plot(TEST_DAYS,diff_ratio_after_recov,"LineWidth",2,"Marker","+","MarkerSize",12)
    hold on
    plot(TEST_DAYS,stem_ratio_after_recov,"LineWidth",2,"Marker","+","MarkerSize",12)
    xticks(TEST_DAYS)
    ax=gca;
    ax.ColorOrder = [1,0,1;1,0,0;0,1,1;0,0,1];
    legend("a.8d diff","a.16d diff","a.8d stem","a.16d stem","Location","north")
    legend("boxoff")

    exportgraphics(h,fullfile(save_dir,"cellTypeProportion.pdf"),...
        "ContentType","image","Resolution",800)
end
