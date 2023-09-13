function Offspring = Preselection(Offspring, PopObj, N, pro, similarity)
    %% Preselection of EPBFS
    
    num = ceil(pro*N); % Number of preselected solutions based on classification error rate
    
    % Selection based on the difference of selected feature ratio
    Next = false(1, size(Offspring, 1));
    index0 = find(similarity ==0);
    index1 = find(similarity ==1);
    randnum = size(index0, 1);
    if randnum >= N-num
        SelectedRatio = mean(Offspring(index0, :), 2);
        [~, sortSelRatio] = sort(SelectedRatio);
        Next(:, index0(sortSelRatio(1:N-num))) = true;
    else
        Next(:, index0) = true;
        neednum = N - num - randnum;
        SelectedRatio = mean(Offspring(index1, :), 2);
        [~, sortSelRatio] = sort(SelectedRatio);
        Next(:, index1(sortSelRatio(1:neednum))) = true;
    end
    
    % Selection based on the predicted classification error rank
    ClassificationError = PopObj(:, 1);
    Potential = find(Next ~= true);
    [~, Rank] = sort(ClassificationError(Potential), 'ascend');
    Next(Potential(Rank(1:num))) = true;
    
    % Preselected offspring
    Offspring = Offspring(Next, :);
end
