clear;
clc;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main program (MATLAB version), for model development & testing purpose only.
% Author: Yi Yang (2301110575@pku.edu.cn)
% Written in MATLAB 2024a, tested on Ubuntu 22.04 & Windows 11
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPORTANT!!!

% Obsolete Note
% This version has been deprecated, although it can still provide some good results, you can 
% simply run this script to get the outputs.
% e.g. matlab main.py

% Replacement
% A more comprehensive, elegant and high-efficient CLI version has been rewritten in Python 
% (see "python" folder), with some new features and customized option extensions. The figures 
% have also been optimized for better visualization.
% Use Python version instead is highly recommanded.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% build saving path
OUTPUT_DIR = make_output_dir();

% load model parameters
[NUM_H3, dyp] = load_default_param();

% model & plot
% commant/uncommant the following lines to plot corresponfing figures

% shematic diagram of model mathmatical principles
plot_schematic_diagram(OUTPUT_DIR)


% plot stable cycle (me0/me3 condition)
plot_epigenetic_stability(32, NUM_H3, dyp, "me0", 40, OUTPUT_DIR)
plot_epigenetic_stability(32, NUM_H3, dyp, "me3", 40, OUTPUT_DIR)


% plot accelerated/decelerated cell division
plot_dynamic_cycle(64, NUM_H3, dyp, 0.5, 12, 30, 0.5, OUTPUT_DIR)
plot_dynamic_cycle(64, NUM_H3, dyp, 3, 12, 30, 0.5, OUTPUT_DIR)


% plot cell division arrest & statistic curves
% major model (act on gene)
plot_division_arrest(256, NUM_H3, dyp, 12, 35, [1,3,5,7,9,11,13,15,17,19], [8,16], 4, true, 0.5, OUTPUT_DIR)
% alternative model (act on chromosome/histone)
plot_division_arrest(256, NUM_H3, dyp, 12, 35, [1,3,5,7,9,11,13,15,17,19], [8,16], 4, false, 0.5, OUTPUT_DIR)


% plot rescue (remove me3/add ATH1) results
plot_rescue_effect(64, NUM_H3, dyp, 20, 30, 5, 0.5, 0.8, 0.3, OUTPUT_DIR)


% k_me - Pdem heatmap
plot_bistability_map(64, NUM_H3, dyp, [11.0,22.0,44.0], 40, 21)



function OUTPUT_DIR = make_output_dir()
    % Description:
    %   Make temporary diretory (date + time) to save output files.

    if ~exist("output","dir")
        mkdir("output");
    end
    
    % Create the temporary directory
    timeStamp = datetime("now","Format", 'yyyyMMddHHmmss');
    OUTPUT_DIR = fullfile('output', string(timeStamp));
    mkdir(OUTPUT_DIR);
end
