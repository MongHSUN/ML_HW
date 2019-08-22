data = readInput('hw3_train.dat');
train_x = data(:,1:2);
train_y = data(:,3);
data2 = readInput('hw3_test.dat');
test_x = data2(:,1:2);
test_y = data2(:,3);
tmp = sort(train_x(:,1));
T = 300;
Eout_gt = [];
alpha = [];
eta = [];
delta = [];
u = [];
initial_u(1,1:100) = 1/100;
u = [u; initial_u];
% adaBoost
thres = preprocessing(train_x);
for t=1:T
    now_u = u(t,:);
    [s, dim, theta, err] = decisionStump(train_x, train_y, now_u, thres);
    eta = [eta, err/sum(now_u)];
    delta = [delta, ((1-eta(t))/eta(t))^0.5];
    alpha = [alpha, log(delta(t))];
    u = [u; updateU(train_x, train_y, s, dim, theta, thres, now_u, delta(t))];
    Eout_gt = [Eout_gt, predict(test_x, test_y, s, dim, theta, thres)];
end
%output
plot([1:T], Eout_gt);
xlabel('t');
ylabel('Eout(gt)');
fprintf('Eout(g1) = %f\n', Eout_gt(1));

function thres = preprocessing(train_x)
    [m, n] = size(train_x);
    thres = zeros(n, m);
    for i=1:n
        tmp = sort(train_x(:,i));
        thres(i,1) = tmp(1) - 1;
        for j=2:m
            thres(i,j) = mean([tmp(j-1), tmp(j)]);
        end
    end
end

function tmp_u = updateU(train_x, train_y, s, dim, theta, thres, now_u, d)
    tmp_u = zeros(1,100);
    tmp_thres = thres(dim, theta);
    for i=1:100
        if train_y(i) ~= s*sign(train_x(i,dim)-tmp_thres)
           tmp_u(i) = now_u(i) * d;
        else
           tmp_u(i) = now_u(i) / d;
        end
    end
    
end

function error = predict(test_x, test_y, s, dim, theta, thres)
    tmp_thres = thres(dim, theta);
    count = 0;
    [m, n] = size(test_x);
    for i=1:m
        if test_y(i) ~= s*sign(test_x(i,dim)-tmp_thres)
            count = count + 1;
        end
    end
    error = count/m;
end

function [s, dim, theta, err] = decisionStump(train_x, train_y, now_u, thres)
    s = 0;
    dim = 0;
    theta = 0;
    err = 100000;
    for i=1:2
        for j=1:100
            errNeg = 0;
            errPos = 0;
            tmp_thres = thres(i,j);
            for k=1:100
                if train_y(k) ~= 1*sign(train_x(k,i)-tmp_thres)
                    errPos = errPos + now_u(k);
                end
                if train_y(k) ~= -1*sign(train_x(k,i)-tmp_thres)
                    errNeg = errNeg + now_u(k);
                end
            end
            if errPos < err
                s = 1;
                dim = i;
                theta = j;
                err = errPos;
            end
            if errNeg < err
                s = -1;
                dim = i;
                theta = j;
                err = errNeg;
            end
        end
    end
end

function data = readInput(fileName)
    formatSpec = '%f %f %f';
    sizeSpec = [3 Inf];
    file = fopen(fileName,'r');
    data = fscanf(file, formatSpec, sizeSpec);
    data = data';
    fclose(file);
end