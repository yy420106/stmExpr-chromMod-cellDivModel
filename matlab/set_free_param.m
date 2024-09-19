function dyp = set_free_param(dyp, varargin)
    % Description
    %   Modify model parameters.
    %
    % Parameters
    %   dyp: struct
    %       Dynamic parameter manager.
    %   varargin
    %       Multiple pairs of key and value for setting. If nothing passed, reset all parameters 
    %       their default values.
    %
    % Returns
    %   dyp: struct
    %       Dynamic parameter manager after change
    
    arguments
        dyp struct 
    end
    arguments (Repeating)
        varargin
    end

    if isempty(varargin)
        run("./parameters.m")
    else
        for i=1:2:length(varargin)
            [key, val] = deal(varargin{i:i + 1});
            assert(ismember(key, ["cell_cycle", "cc0","mu", "alpha_lim", "alpha_expk", "epsilon", ...
                "sigma", "Kd", "f_min", "f_max", "f_lim", "Pt", "gamma_transcr", "prot_per_transcr", ...
                "kappa", "beta", "e_distal", "rho", "k_me", "Pdem", "Pex", "A", "B", "omega_lim"]), "Invalid parameter!")
            if strcmp(key, "k_me")
                dyp.(key) = val;
                dyp.k_me01 = 9 * dyp.k_me;
                dyp.k_me12 = 6 * dyp.k_me;
                dyp.k_me23 = dyp.k_me;
                dyp.gamma_me01 = dyp.k_me01 / 20;
                dyp.gamma_me12 = dyp.k_me12 / 20;
                dyp.gamma_me23 = dyp.k_me23 / 20;
            elseif strcmp(key, "f_min") || strcmp(key, "Pdem")
                dyp.(key) = val;
                dyp.gamma_dem = dyp.f_min * dyp.Pdem;
            else
                dyp.(key) = val;
            end
        end
    end
end

