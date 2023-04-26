function Offspring = OffspringGeneration(Parent)
    
    %% Parameter setting
    [proC, proM] = deal(1, 1);
    
    if isa(Parent(1), 'SOLUTION')
        Parent = Parent.decs;
    end

    Parent1 = Parent(1:floor(end/2), :);
    Parent2 = Parent(floor(end/2) + 1:floor(end/2)*2, :);
    [N, D]  = size(Parent1);
    
    %% Genetic operators for binary encoding
    % One point crossover
    k = repmat(1:D, N, 1) > repmat(randi(D, N, 1), 1, D);
    k(repmat(rand(N,1) > proC, 1, D)) = false;
    Offspring1    = Parent1;
    Offspring2    = Parent2;
    Offspring1(k) = Parent2(k);
    Offspring2(k) = Parent1(k);
    Offspring     = [Offspring1; Offspring2];
    % Bit-flip mutation
    Site = rand(2*N, D) < proM/D;
    Offspring(Site) = ~Offspring(Site);

    % Repair solutions that do not select any features
    flag = sum(Offspring, 2) == 0;
    if sum(flag, 1) > 0
        Offspring(flag, 1:end) = randi([0, 1], sum(flag, 1), D);
    end
        
    % repair duplicated solutions
    boolis = ismember(Offspring, Parent, 'rows');
    normal = Offspring(boolis==0, 1:end);
    duplic = Offspring(boolis==1, 1:end);
    for i =1:size(duplic, 1)
        index1 = find( duplic(i, :));
        index2 = find(~duplic(i, :));
        if size(index1, 2) > 0
            duplic(i, index1(randi(end, 1, 1))) = 0;
        end
        if size(index2, 2) > 0
            duplic(i, index2(randi(end, 1, 1))) = 1;
        end
    end
    Offspring = [normal; duplic];
        
    % get unique offspring and individuals (function evaluated)
    Offspring = unique(Offspring, 'rows');
    Offspring = Offspring(sum(Offspring, 2)>0, 1:end);
end