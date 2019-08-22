data = readInput('hw2_lssvm_all.dat');
trainX = data(1:400,1:11);
trainY = data(1:400,12);
testX = data(401:500,1:11);
testY = data(401:500,12);
lambda = [0.01,0.1,1,10,100];
%kernel ridge regression
Ein = zeros(3,1);
for i=1:5
    l = lambda(i);
    beta = computeBeta(trainX, trainY, l);
    Ein(i) = computeError(beta, testX, testY, trainX);
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

function beta = computeBeta(dataX, dataY, lambda)
    [m, n] = size(dataX);
    k = zeros(m,m);
    for i=1:m
        for j=1:m
            data1 = dataX(i,:);
            data2 = dataX(j,:);
            k(i,j) = dot(data1, data2);
        end
    end
    beta = inv(lambda*eye(m)+k)*dataY;
end

function error = computeError(beta, dataX, dataY, trainX)
    [m, n] = size(dataX);
    count = 0;
    for i=1:m
       x = dataX(i,:);
       k = zeros(400,1);
       for j=1:400
           X = trainX(j,:);
           k(j,1) = dot(x,X);
       end
       if sign(dot(beta,k))~=dataY(i)
           count = count + 1;
       end
    end
    error = count / m;
end