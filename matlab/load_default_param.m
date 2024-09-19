function [NUM_H3, dyp] = load_default_param()
    % Description:
    %   Load model parameters with their default values.
    %
    % Returns:
    %   NUM_H3: int
    %       Number of H3 histones, adjusted to STM gene length.
    %   dyp: struct
    %       Dynamic parameter manager that includes the following keys: cell_cycle, cc0, mu, alpha_lim, 
    %       alpha_expk, epsilon, sigma, Kd, f_min, f_max, f_lim, Pt, gamma_transcr, prot_per_transcr, kappa, 
    %       beta, e_distal, rho, k_me, Pdem, Pex, A, B, omega_lim, k_me01, k_me12, k_me23, gamma_me01, 
    %       gamma_me12, gamma_me23, gamma_dem.
    
    % static parameter
    NUM_H3 = 2 * ceil(3482 / 200);  % STM gene length: 3482 bp

    % dynamic paramter (dyp)
    dyp = struct();

    % cell division
    dyp.cell_cycle = 22.0;
    % gene activation
    dyp.cc0 = 22.0;
    dyp.mu = 0.5;
    dyp.alpha_lim = 1e-2;
    dyp.alpha_expk = log(2) / 22.0;
    dyp.epsilon = 1.0;
    dyp.sigma = 2.0;
    dyp.Kd = 180.0;
    % gene transcription
    dyp.f_min = 1e-4;
    dyp.f_max = 4e-3;
    dyp.f_lim = 1 / 60;
    dyp.Pt = 1 / 3;
    dyp.gamma_transcr = 9e-8;
    % protein production & degradation
    dyp.prot_per_transcr = 1.0;
    dyp.kappa = 4e-6;
    % histone methylation
    dyp.beta = 1.0;
    dyp.e_distal = 0.001;
    dyp.rho = 1 / 10;
    dyp.k_me = 8e-6;
    % histone demethylation & exchange
    dyp.Pdem = 5e-3;
    dyp.Pex = 1.5e-3;
    % indirect interaction (alternative)
    dyp.A = 0.5;
    dyp.B = 11.0 + log(1.5);
    dyp.omega_lim = 2.5;

    % dependend
    dyp.k_me01 = 9 * dyp.k_me;
    dyp.k_me12 = 6 * dyp.k_me;
    dyp.k_me23 = dyp.k_me;
    dyp.gamma_me01 = dyp.k_me01 / 20;
    dyp.gamma_me12 = dyp.k_me12 / 20;
    dyp.gamma_me23 = dyp.k_me23 / 20;
    dyp.gamma_dem = dyp.f_min * dyp.Pdem;
end
