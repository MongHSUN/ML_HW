data = readInput('hw2_lssvm_all.dat');
trainX = data(1:400,1:10);
trainY = data(1:400,11);
gamma = [32,2,0.125];
lambda = [0.001,1,1000];
%kernel ridge regression
Ein = zeros(3,3);
for i=1:3
    for j=1:3
        g = gamma(i);
        l = lambda(j);
        beta = computeBeta(trainX, trainY, l, g);
        Ein(i,j) = computeError(beta, g, trainX, trainY);
    end
end
%output result
for i=1:3
    for j=1:3
        fprintf('gamma = %f, lambda = %f, Ein = %f\n', gamma(i), lambda(j), Ein(i,j));
    end
end

function data = readInput(fileName)
    formatSpec = '%f %f %f %f %f %f %f %f %f %f %f';
    sizeSpec = [11 Inf];
    file = fopen(fileName,'r');
    data = fscanf(file, formatSpec, sizeSpec);
    data = data';
    fclose(file);
end

function beta = computeBeta(dataX, dataY, lambda, gamma)
    [m, n] = size(dataX);
    k = zeros(m,m);
    for i=1:m
        for j=1:m
            data1 = dataX(i,:);
            data2 = dataX(j,:);
            k(i,j) = exp(-gamma*dot((data1-data2),(data1-data2)));
        end
    end
    beta = inv(lambda*eye(m)+k)*dataY;
end

function error = computeError(beta, gamma, dataX, dataY)
    [m, n] = size(dataX);
    count = 0;
    for i=1:m
       x = dataX(i,:);
       k = zeros(m,1);
       for j=1:m
           X = dataX(j,:);
           k(j,1) = exp(-gamma*dot((x-X),(x-X)));
       end
       if sign(dot(beta,k))~=dataY(i)
           count = count + 1;
       end
    end
    error = count / m;
end