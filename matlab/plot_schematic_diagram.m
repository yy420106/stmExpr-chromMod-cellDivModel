function plot_schematic_diagram(save_dir)
    % Description:
    %   Plot schematic diagram to explain the mathematical principles.
    %
    % Parameters:
    %   save_dir: string (default=pwd)
    %       Directory path for saving.
    
    arguments
        save_dir string = string(pwd)
    end

    % example data (with no real meaning) to illustrate principlrs
    x = linspace(0,30,301);
    y0 = 5*x;  % degredation rate
    yt = 125*x.^2./(100+x.^2);  % production rate (upper bound)
    yb = 50*x.^2./(100+x.^2);  % production rate (lower bound)
    
    % plot
    f = figure('Name',"SchematicDiagram");
    hold on

    k_array = linspace(50,125,151);
    c=[linspace(0,1,75)',linspace(0,1,75)',ones(75,1)];c=[c;1,1,1;rot90(c,2)];c=[c,0.3*ones(151,1)];
    for k=1:length(k_array)
        yk = k_array(k)*x.^2./(100+x.^2);  % production rate (intermediate state)
        plot(x,yk,"LineWidth",1,"Color",c(k,:))  % gradient color
    end
    
    xline(5,"--","LineWidth",1)
    xline(20,"--","LineWidth",1)   
    l1=plot(x,y0,"Color","green","LineWidth",2);
    l2=plot(x,yt,"Color","red","LineWidth",2);
    l3=plot(x,yb,"Color","blue","LineWidth",2);  
    s1 = scatter(5,25,50,"black","filled","Marker","o");  % critical point
    s2 = scatter(20,100,50,"black","filled","Marker","^");  % stable point   
    xticks([]);xlabel("Protein Number")
    yticks([]);ylabel("Propensity")
    legend([l1,l2,l3,s1,s2],{"Degradation","Production (me0)","Production (me3)",...
        "Metastable Point","Stable Point"},"Location","best")

    % save
    exportgraphics(f,fullfile(save_dir,"schematicDiagram.pdf"),...
        "ContentType","vector","Resolution",1000)
end
