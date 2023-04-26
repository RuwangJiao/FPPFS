function [mi, su, Ha, Hb] = CalInformationTheoreticMeasures(a, b)
    %% Calculate information theoretic based measures
    % Calculate the overlap
    [Ma, Na] = size(a);
    [Mb, Nb] = size(b);
    M        = min(Ma, Mb);
    N        = min(Na, Nb);

    % Initialize histogram array
    hab = zeros(256, 256);
    ha  = zeros(1, 256);
    hb  = zeros(1, 256);

    % Normalization
    if max(max(a)) ~= min(min(a))
        a = (a - min(min(a)))/(max(max(a)) - min(min(a)));
    else
        a = zeros(M, N);
    end

    if max(max(b)) - min(min(b))
        b = (b - min(min(b)))/(max(max(b)) - min(min(b)));
    else
        b = zeros(M, N);
    end

    a = double(int16(a*255)) + 1;
    b = double(int16(b*255)) + 1;

    % Statistical histogram
    for i=1:M
        for j=1:N
            indexx = a(i, j);
            indexy = b(i, j) ;
            hab(indexx, indexy) = hab(indexx, indexy) + 1; % Joint histogram
            ha(indexx) = ha(indexx) + 1; % Histogram of a
            hb(indexy) = hb(indexy) + 1; % Histogram of b
        end
    end

    % Joint entropy of a and b
    hsum = sum(sum(hab));
    index = find(hab~=0);
    p = hab/hsum;
    Hab = sum(sum(-p(index).*log(p(index)))) + 1e-6;

    %Entropy of a
    hsum = sum(sum(ha));
    index = find(ha~=0);
    p = ha/hsum;
    Ha = sum(sum(-p(index).*log(p(index)))) + 1e-6;

    % Entropy of b
    hsum = sum(sum(hb));
    index = find(hb ~= 0);
    p = hb/hsum;
    Hb = sum(sum(-p(index).*log(p(index)))) + 1e-6;

    % Mutual information (a, b)
    mi = Ha + Hb - Hab + 1e-6;

    % Symmetrical uncertainty (a, b)
    su = 2*(mi/(Ha+Hb)) + 1e-6;
end