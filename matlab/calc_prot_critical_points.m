function prot_critical_points = calc_prot_critical_points(transcr_state, dyp, evalf_n, act_on_gene, p0)
    % Description:    
    %   Compute the critical number of protein molecules for the production-degradation system to be stable or
    %   metastable at different chromatin repression state.
    %
    % Parameters
    %   transcr_state: string ("max","min")
    %       Transcription state of STM gene.'max'/'min' means the ratio of inhibitory modification within gene 
    %       locus equals 0/1, respectively.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    %   evalf_n: int
    %       Precision of function evaluation.
    %   act_on_gene: logical
    %       Whether to apply the master model (directly act on gene transcription) or alternative model 
    %       (indirectly act on gene transcription by epigentic modification).
    %   p0: float array (default=[0,0.0001,0.001,0.01,0.1,1,10,100,1000,10000])
    %       Initial points (prediction roots) used in numerical solver.
    %
    % Returns
    %   prot_crtitical_points: float array, size [1,:]
    %       Array of possible crtical points (ascending order), with undetermined length.
        
    arguments
        transcr_state string {mustBeMember(transcr_state,{'max','min'})}
        dyp struct
        evalf_n double
        act_on_gene logical
        p0 (1,:) double = [0, 10.^(-4:4)]
    end
    
    % intermediate variable
    if act_on_gene
        alpha = max((dyp.cc0 / dyp.cell_cycle) ^ dyp.mu, dyp.alpha_lim);
    else
        alpha = 1.0;
    end
    
    % build dynamic function & initial points
    if strcmp(transcr_state, "max")
        lim = alpha * dyp.epsilon * dyp.f_max;
        prot_dynamic_func = @(p)(dyp.prot_per_transcr*(...
            alpha*dyp.epsilon*p^dyp.sigma/(dyp.Kd+p^dyp.sigma)*dyp.f_max+dyp.gamma_transcr)-dyp.kappa*p);
    elseif strcmp(transcr_state, "min")
        lim = alpha * dyp.epsilon * dyp.f_min;
        prot_dynamic_func = @(p)(dyp.prot_per_transcr*(...
            alpha*dyp.epsilon*p^dyp.sigma/(dyp.Kd+p^dyp.sigma)*dyp.f_min+dyp.gamma_transcr)-dyp.kappa*p);
    else
        error("Error. Invaild input for 'transcr_state'.")
    end
    
    % solve equation
    option = optimoptions("fsolve","OptimalityTolerance",1e-16,"MaxFunctionEvaluations",400,"MaxIterations",1000);
    root = arrayfun(@(p)fsolve(prot_dynamic_func, p, option), p0);
    valid_root = real(root(real(root)>=0&imag(root)==0));  % keep only positive real number
    
    % extract valid value
    lim_root = dyp.prot_per_transcr*(dyp.f_lim +dyp.gamma_transcr)/dyp.kappa;
    if lim <= dyp.f_lim
        prot_critical_points = sort(unique(round(valid_root, evalf_n)));
    else
        split_point = dyp.Kd * (dyp.f_lim/lim) / (1 - dyp.f_lim/lim) ^ (1/sigma);
        prot_critical_points = sort(unique(round(valid_root(valid_root<=split_point), evalf_n)));
        if lim_root > split_point
            prot_critical_points = [prot_critical_points, round(lim_root, evalf_n)];
        end
    end
end
