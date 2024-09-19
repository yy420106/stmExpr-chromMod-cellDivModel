function plot_bistability_map(N_SAMPLES, NUM_H3, dyp, TEST_CELL_CYCLE, MAP_CYCLE, MAPSIZE, ...
    K_ME_LOG_RANGE, PDEM_LOG_RANGE, save_dir)
    % Description:
    %   Plot k_me - Pdem bistability map under different cell cycle conditions.
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
    %   TEST_CELL_CYCLE: float array
    %       Cell cycle for testing.
    %   MAP_CYCLE: int
    %       Number of cycle to be performed to collect data.
    %   MAPSIZE: int array
    %       Pixels of each dimension.
    %   K_ME_LOG_RANGE: float array (default=[-6, -2])
    %       Log10 range of model paramter 'k_me'.
    %   PDEM_LOG_RANGE: float array (default=[-4, 0])
    %       Log10 range of model paramter 'Pdem'.
    %   save_dir: string (default=pwd)
    %       Directory path for saving.

    arguments
        N_SAMPLES double
        NUM_H3 double
        dyp struct
        TEST_CELL_CYCLE (1,:) double
        MAP_CYCLE double
        MAPSIZE double
        K_ME_LOG_RANGE (1,2) double = [-6,-2]
        PDEM_LOG_RANGE (1,2) double = [-4,0]
        save_dir string = string(pwd)
    end

    % sample number of parameters
    k_me_ticks = logspace(K_ME_LOG_RANGE(1), K_ME_LOG_RANGE(2), MAPSIZE);
    Pdem_ticks = logspace(PDEM_LOG_RANGE(1), PDEM_LOG_RANGE(2), MAPSIZE);

    % compute bistability for each parameter set
    for cc_idx=1:length(TEST_CELL_CYCLE)
        dyp_cc = dyp;
        dyp_cc = set_free_param(dyp_cc,"cell_cycle",TEST_CELL_CYCLE(cc_idx));
        prot_critical_points = calc_prot_critical_points("max",dyp_cc,4,true);
        cp_max = prot_critical_points(end);
        
        % initialize output
        bistabilityMap = zeros(MAPSIZE, MAPSIZE);
        
        wb = waitbar(0,['Processing ... finished: ',num2str(0),'/',num2str(MAPSIZE^2)]);
        for i=1:MAPSIZE
            for j=1:MAPSIZE
                % build model
                geneExpr0 = [round(cp_max)*ones(1,N_SAMPLES),zeros(1,N_SAMPLES)];
                meState0 = [zeros(N_SAMPLES, NUM_H3); 3*ones(N_SAMPLES,NUM_H3)];
                time_to_next_repl0 = zeros(1, 2*N_SAMPLES);
                time_records = linspace(0, MAP_CYCLE*dyp_cc.cell_cycle, MAP_CYCLE*dyp_cc.cell_cycle*60+1);
                dyp_cc = set_free_param(dyp_cc,"k_me",k_me_ticks(i),"Pdem",Pdem_ticks(j));
                [~, samples_meState, ~, ~] = gillespie_ssa_parallel(...
                    geneExpr0, meState0, time_to_next_repl0, dyp_cc, time_records, true, true, ...
                    max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim));
                
                % compute bistability
                to_next_repl = ceil(time_records/dyp_cc.cell_cycle)*dyp_cc.cell_cycle-time_records;
                valid_samples_meState = samples_meState(:,(to_next_repl>=0)&(to_next_repl<1),:);
                samples_me23_ratio = sum(valid_samples_meState>=2,3)/NUM_H3;
                samples_Pon = sum(samples_me23_ratio<dyp.Pt/4,2)/size(samples_me23_ratio,2);
                samples_Poff = sum(samples_me23_ratio>3*dyp.Pt/4,2)/size(samples_me23_ratio,2);

                bistabilityMap(i,j) = 4*mean(samples_Pon)*mean(samples_Poff);
                waitbar(((i-1)*MAPSIZE+j)/MAPSIZE^2, wb, ...
                    ['Processing ... finished: ',num2str(((i-1)*MAPSIZE+j)),'/',num2str(MAPSIZE^2)])
            end
        end
        close(wb)

        % plot
        f=figure("Name",strcat("biMapofCellCycle",num2str(round(dyp_cc.cell_cycle,1)),"h"));
        hm = heatmap(flipud(bistabilityMap));
        
        hm.Interpreter = "latex";
        hm.XLabel = "$\lg P_{dem}$";
        hm.YLabel = "$\lg k_{me}$";
        hm.XDisplayLabels = linspace(PDEM_LOG_RANGE(1), PDEM_LOG_RANGE(2), MAPSIZE);
        hm.YDisplayLabels = flip(linspace(K_ME_LOG_RANGE(1), K_ME_LOG_RANGE(2), MAPSIZE));
        hm.CellLabelColor = "none";
        hm.Colormap = copper;
        hm.GridVisible = "off";

        exportgraphics(f,fullfile(save_dir,...
            strcat("biMapofCellCycle",num2str(round(dyp_cc.cell_cycle,1)),"h.pdf")),...
            "ContentType","image","Resolution",800)
    end 
end
