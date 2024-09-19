function new_time_to_next_repl = calc_time_to_next_repl_after_ev( ...
    time_to_next_repl, cell_cycle, ev_time)
    % Calculate the time to next DNA replication after model evolution with constant cell cycle.
    %
    % Parameters:
    %   time_to_next_repl: float array, size [1, #]
    %       Time to next DNA replication in each sample before model evolution.
    %   cell_cycle: float
    %       Cell cycle of the model.
    %   ev_time: float
    %       Evolution time (unit: hour).
    %
    % Returns:
    %   new_time_to_next_repl: float array, size [1, #]
    %       Time to next DNA replication after evolution.
    
    arguments
        time_to_next_repl (1,:) double
        cell_cycle double
        ev_time double
    end

    if isinf(cell_cycle)
        new_time_to_next_repl = inf(size(time_to_next_repl));
    else
        assert(all(time_to_next_repl < cell_cycle));
        new_time_to_next_repl = time_to_next_repl + ceil((ev_time - time_to_next_repl) / cell_cycle) * cell_cycle - ev_time;
    end
end
