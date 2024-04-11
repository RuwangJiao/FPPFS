classdef FPPFS < ALGORITHM
% <multi> <binary> <constrained/none> 
% A Filter-based Performance Predictor for Multiobjective Feature Selection
%------------------------------- Reference --------------------------------
% R. Jiao, B. Xue, M. Zhang. Learning to Preselection: A Filter-based Performance
% Predictor for Multiobjective Feature Selection in Classification. IEEE 
Transactions on Evolutionary Computation, 2024, doi: 10.1109/TEVC.2024.3373802.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Setting population size and maxFE for fair comparison %%
            %[Problem.N, Problem.maxFE] = InitialExperimentSetting(Problem);
            
            % Calculate correlations
            [MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR] = InitializeCorrelations(Problem);
            
            % Generate an initial population
            Population = InitializePopulation(Problem);

            % Environmental selection
            [Population, FrontNo, CrowdDis] = EnvironmentalSelection(Population, Problem.N);

            % Initialize the training set for the performance predictor
            TrainingSet = Population;
            L = [];

            while Algorithm.NotTerminated(Population)
                % Calculate classification error rank in training set
                [realRank, approximateRank] = CalculateClassificationErrorRank(TrainingSet, MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR);

                % Calculate weights
                [max_w, index_w, pro] = CalculateWeights(realRank, approximateRank);
                L = [L, max_w];

                % Mating selection
                MatingPool = TournamentSelection(2, 5*Problem.N, FrontNo, -CrowdDis);

                % Offspring reproduction
                Offspring  = OffspringGeneration(Population(MatingPool));

                % Similarity of selected feature ratio
                similarity = CalculateSimilarity(Offspring, Population);

                % Predicting classification error via performance predictor
                PredictRank = PredictClassificationErrorRank(Offspring, MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR, index_w);

                % Preselection
                Offspring = Preselection(Offspring, [PredictRank, sum(Offspring, 2)], Problem.N, pro, similarity);

                % Evaluation for each solution
                Offspring = SOLUTION(Offspring);

                % Training set update for the performance predictor
                TrainingSet = UpdateTrainingSet([TrainingSet, Offspring], Problem.N);

                % Environmental selection
                [Population, FrontNo, CrowdDis] = EnvironmentalSelection([Population, Offspring], Problem.N);

                %%%%% Applied to the test set %%%%%
                %Population = FSTraining2Test(Problem, Population);
            end
        end
    end
end

function Population = InitializePopulation(Problem)
    T = min(Problem.D, Problem.N * 3);
    Pop = zeros(Problem.N, Problem.D);
    for i = 1 : Problem.N
        k = randperm(T, 1);
        j = randperm(Problem.D, k);
        Pop(i, j) = 1;
    end
    Population = SOLUTION(Pop);
end

function [MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR] = InitializeCorrelations(Problem)
    %% Calculate correlations
    MI_fc = zeros(1, Problem.D);        % Mutual information between features and labels
    SU_fc = zeros(1, Problem.D);        % Symmetrical uncertainty between features and labels
    H     = zeros(1, Problem.D);        % Entroy of each feature
    MI_ff = ones(Problem.D, Problem.D); % Mutual information between features
    SU_ff = ones(Problem.D, Problem.D); % Symmetrical uncertainty between features
    [MI_cc, ~, ~, ~] = CalInformationTheoreticMeasures(Problem.TrainY, Problem.TrainY);
    for i = 1:Problem.D
        [MI_fc(i), SU_fc(i), ~, HC] = CalInformationTheoreticMeasures(Problem.TrainX(:,i), Problem.TrainY);
         for j = i+1:Problem.D
            [MI_ff(i, j), SU_ff(i, j), H(i), H(j)] = CalInformationTheoreticMeasures(Problem.TrainX(:,i), Problem.TrainX(:,j));
            MI_ff(j, i) = MI_ff(i, j);
            SU_ff(j, i) = SU_ff(i, j);
         end
    end
    DIV = MI_fc./H;
    IH = ones(Problem.D, Problem.D);
    for i = 1:Problem.D
        for j = i:Problem.D
            IH(i, j) = DIV(i) + DIV(j);
            IH(j, i) = IH(i, j);
        end
    end
    RedCR = IH.*MI_ff;
    IHC = zeros(1, Problem.D);
    MinI = zeros(Problem.D, Problem.D);
    MaxI = zeros(Problem.D, Problem.D);
    for i = 1:Problem.D
        IHC(i) = min(H(i), HC);
        for j = i:Problem.D
            MinI(i, j) = min(MI_fc(i), MI_fc(j));
            MinI(j, i) = MinI(i, j);
            MaxI(i, j) = max([MI_ff(i, j), MI_fc(i), MI_fc(j)]);
            MaxI(j, i) = MaxI(i, j);
        end
    end
   RelNor = MI_fc/MI_cc;                    % Normalized relevance
   RedNor = MI_ff./MaxI.*MinI;
   RedNor = RedNor - diag(diag(RedNor));  % Normalized redundancy
end

function similarity = CalculateSimilarity(Offspring, Population)
%% The similarity between offspring and parent populations in terms of
%% number of selected features
    similarity = zeros(size(Offspring, 1), 1);
    Obj = Population.objs;
    Obj2 = Obj(:, 2);
    for i = 1:size(Offspring, 1)
        selenum = sum(Offspring(i, :))./size(Offspring, 2);
        index = selenum == Obj2;
        index = max(index);
        similarity(i, :) = index;
    end
end

function [realRank, approximateRank] = CalculateClassificationErrorRank(TrainingData, MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR)
    %% Calculate classification error rank in the training set
    PopObj = TrainingData.objs;
    PopDec = logical(TrainingData.decs);
    [~, Rank1] = sort(PopObj(:,1));
    [~, realRank] = sort(Rank1);
    PerformancePredictor = zeros(size(PopObj,1), 5);
    for i = 1 : size(PopObj, 1)
        k = sum(PopDec(i, :));
        Rel = sum(MI_fc(:, PopDec(i, :)), 2);
        Red = sum(sum(MI_ff(PopDec(i, :), PopDec(i, :))));
        REDNOR = sum(max(RedNor(PopDec(i, :), PopDec(i, :)))/MI_cc);
        RELNOR = sum(RelNor(:, PopDec(i, :)), 2);
        Relsu = sum(SU_fc(:, PopDec(i, :)), 2)./k;
        Redsu = sum(sum(SU_ff(PopDec(i, :), PopDec(i, :))))./(k*k);
        REDCR = sum(sum(RedCR(PopDec(i, :), PopDec(i, :))));
        PerformancePredictor(i, 1) = Rel./k;                          % Mean relevance
        PerformancePredictor(i, 2) = Rel./k - Red./(k*k);             % MRMR
        PerformancePredictor(i, 3) = Rel./k-0.5*REDCR./(k*k);         % MIFS-CR
        PerformancePredictor(i, 4)  =  RELNOR./k - REDNOR./(k*k);     % N-MRMCR-MI
        PerformancePredictor(i, 5) = k*Relsu./(sqrt(k+k*(k-1)*Redsu));% CFS
    end 
    [~, Rank] = sort(PerformancePredictor, 'descend');
    [~, approximateRank] = sort(Rank);
end

function PredictRank = PredictClassificationErrorRank(Offspring, MI_fc, MI_ff, SU_fc, SU_ff, MI_cc, RedNor, RelNor, RedCR, index_w)
    %% Predict classification error rank of each solution
    OffDec = logical(Offspring);
    PerformancePredictor = zeros(size(OffDec, 1), 1);
    for i = 1 : size(OffDec, 1)
        k = sum(OffDec(i, :));
        switch(index_w)
            case 1   % Mean relevance
                Rel = sum(MI_fc(:, OffDec(i, :)), 2);
                PerformancePredictor(i, :) = Rel./k; 
            case 2   % MRMR
                Rel = sum(MI_fc(:, OffDec(i, :)), 2);
                Red = sum(sum(MI_ff(OffDec(i, :), OffDec(i, :))));
                PerformancePredictor(i, :) = Rel./k - Red./(k*k); 
            case 3   % MIFSCR
                Rel = sum(MI_fc(:, OffDec(i, :)), 2);
                REDCR = sum(sum(RedCR(OffDec(i, :), OffDec(i, :))));
                PerformancePredictor(i, :) = Rel./k-0.5*REDCR./(k*k);
            case 4   % N-MRMCR-MI
                REDNOR = sum(max(RedNor(OffDec(i, :), OffDec(i, :)))/MI_cc);
                RELNOR = sum(RelNor(:, OffDec(i, :)), 2);
                PerformancePredictor(i, :)  =  RELNOR./k - REDNOR./(k*k);
            case 5   % CFS
                Relsu = sum(SU_fc(:, OffDec(i, :)), 2)./k;
                Redsu = sum(sum(SU_ff(OffDec(i, :), OffDec(i, :))))./(k*k);
                PerformancePredictor(i, :) = k*Relsu./(sqrt(k+k*(k-1)*Redsu));
        end
    end 
    [~, Rank] = sort(PerformancePredictor, 'descend');
    [~, PredictRank] = sort(Rank);
end

function [max_w, index_w, pro] = CalculateWeights(F, Fpre)
    %% Calculate the weight and the selection probability
    w = zeros(size(Fpre, 2), 1);
    for i=1:size(Fpre, 2)
        w(i, :) = corr(F, Fpre(:, i), 'type', 'Spearman');
    end
    % Majority voting
    [max_w, index_w] = max(w);
    if max_w < 0 
        max_w = 0;
    end
    % Calculate the selection proportion
    pro = CorrelationBasedProbability(max_w);
end

function pro = CorrelationBasedProbability(w)
    %% Calculate the selection probability based on the correlation
    cp       = 5;
    z        = 1e-8;
    Nearzero = 1e-15;
    B        = 1./power(log((1 + z)./z), 1.0./cp);
    B(B==0)  = B(B==0) + Nearzero;
    f        = exp( -(w./B).^cp );
    tmp      = find(abs(f-z) < Nearzero);
    f(tmp)   = f(tmp).*0 + z;
    pro      = 1 - f + z;
    pro(pro<=0) = 0;
end

function TrainingSet = UpdateTrainingSet(TrainingSet, N)
    %% Update the Training Set
    DataDec = TrainingSet.decs;
    [~, index] = unique(DataDec, 'rows');
    TrainingSet = TrainingSet(index);
    if size(TrainingSet, 2) > 5*N
        TrainingSet = TrainingSet(:, end-5*N+1:end);
    end
end
