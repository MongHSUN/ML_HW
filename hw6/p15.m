data = readInput('hw2_lssvm_all.dat');
train = data(1:400,:);
lambda = [0.01,0.1,1,10,100];
t = 250;
%linear ridge regression
[m, n] = size(train);
Ein = [];
for i=1:5
    vote = zeros(400,1);
    l = lambda(i);
    for j=1:t
        tmp = bagging(train);
        tmpX = tmp(:,1:11);
        tmpY = tmp(:,12);
        w = inv(l*eye(11)+tmpX'*tmpX)*tmpX'*tmpY;
        for k=1:m
            x = train(k, 1:11);
            vote(k) = vote(k) + sign(dot(x,w));
        end
    end
    Ein = [Ein, computeError(vote, train(:,12))];
end
%output result
for i=1:5
    fprintf('lambda = %f, Ein = %f\n', lambda(i), Ein(i));
end

function data = readInput(fileName)
    formatSpec = '%f %f %f %f %f %f %f %f %f %f %f';
    sizeSpec = [11 Inf];
    file = fopen(fileName,'r');
    data = fscanf(file, formatSpec, sizeSpec);
    data = data';
    [m, n] = size(data);
    one = ones(m,1);
    data = [one data];
    fclose(file);
end

function output = bagging(train)
    output = [];
    r = unidrnd(400,400,1);
    for i=1:400
        output = [output; train(r(i),:)];
    end
end

function error = computeError(vote, dataY)
    [m, n] = size(dataY);
    count = 0;
    for i=1:m
        if sign(vote(i))~=dataY(i)
            count = count + 1;
        end
    end
    error = count / m;
end