function Offspring = Preselection(Offspring, PopObj, N, pro, similarity)
    %% Preselection of EPBFS
    
    num = ceil(pro*N); % Number of preselected solutions based on classification error rate
    
    % Selection based on the difference of selected feature ratio
    Next = false(1, size(Offspring, 1));
    index0 = find(similarity ==0);
    index1 = find(similarity ==1);
    randnum = size(index0, 1);
    if randnum >= N-num
        randomSelect = randperm(randnum, N - num);
        Next(:, index0(randomSelect)) = true;
    else
        Next(:, index0) = true;
        neednum = N - num - randnum;
        randomSelect = randperm(N-randnum, neednum);
        Next(:, index1(randomSelect)) = true;
    end
    
    % Selection based on the predicted classification error rank
    ClassificationError = PopObj(:, 1);
    Potential = find(Next ~= true);
    [~, Rank] = sort(ClassificationError(Potential), 'ascend');
    Next(Potential(Rank(1:num))) = true;
    
    % Preselected offspring
    Offspring = Offspring(Next, :);
end