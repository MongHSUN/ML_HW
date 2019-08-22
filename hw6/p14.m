data = readInput('hw2_lssvm_all.dat');
trainX = data(1:400,1:11);
trainY = data(1:400,12);
testX = data(401:500,1:11);
testY = data(401:500,12);
lambda = [0.01,0.1,1,10,100];
%linear ridge regression
[m, n] = size(trainX);
Eout = [];
for i=1:5
    l = lambda(i);
    w = inv(l*eye(n)+trainX'*trainX)*trainX'*trainY;
    Eout = [Eout, computeError(w, testX, testY)];
end
%output result
for i=1:5
    fprintf('lambda = %f, Eout = %f\n', lambda(i), Eout(i));
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

function error = computeError(w, dataX, dataY)
    [m, n] = size(dataX);
    count = 0;
    for i=1:m
       x = dataX(i,:);
       if sign(dot(x,w))~=dataY(i)
           count = count + 1;
       end
    end
    error = count / m;
end