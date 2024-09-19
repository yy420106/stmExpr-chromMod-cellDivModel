function new_time_to_next_repl = calc_time_to_next_repl_after_ccc( ...
    old_time_to_next_repl, old_cell_cycle, new_cell_cycle)
    % Calculate the time to next DNA replication after cell cycle changes based on equal-scale transformation.
    %
    % Parameters:
    %   old_time_to_next_repl: float array, size [1, #]
    %       Time to next DNA replication before cell cycle changes.
    %   ole_cell_cycle: float
    %       Old cell cycle.
    %   new_cell_cycle: float
    %       New cell cycle.
    %
    % Returns:
    %   new_time_to_next_repl: float array, size [1, #]
    %       Time to next DNA replication after cell cycle changes.
    %
    % Notes:
    %   If cell cycle is infinite, it represents that cell stops dividing. If a cell restart division from
    %   quiescence, 'new_time_to_next_repl' is set to 0. If 'old_time_to_next_repl' is 0.0, the cell will immediately
    %   do a DNA replication and divide the next moment before new cycle is applied.
    
    arguments
        old_time_to_next_repl (1,:) double
        old_cell_cycle double
        new_cell_cycle double
    end

    if isinf(old_cell_cycle)
        if isinf(new_cell_cycle)
            new_time_to_next_repl = inf(size(old_time_to_next_repl));
        else
            new_time_to_next_repl = zeros(size(old_time_to_next_repl));
        end
    else
        assert(all(old_time_to_next_repl < old_cell_cycle));
        
        % equal-scale transform
        % special treatment for NaN
        % if 'old_time_to_next_repl'=0.0 and 'new_cell_cycle'=inf, set 'new_time_to_next_repl'=0.0
        new_time_to_next_repl = old_time_to_next_repl ./ old_cell_cycle .* new_cell_cycle;
        new_time_to_next_repl(isnan(new_time_to_next_repl)) = 0;
    end
end
