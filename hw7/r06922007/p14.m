data = readInput('hw3_train.dat');
train_x = data(:,1:2);
train_y = data(:,3);
tmp = sort(train_x(:,1));
T = 300;
Ein_gt = [];
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
    Ein_gt = [Ein_gt, predict(train_x, train_y, s, dim, theta, thres)];
end
sum_u = [];
for t=1:T
   sum_u = [sum_u, sum(u(t,:))];
end
%output
plot([1:T], sum_u);
xlabel('t');
ylabel('Ut');
fprintf('U_2 = %f\n', sum_u(2));
fprintf('U_T = %f\n', sum_u(T))

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

function error = predict(train_x, train_y, s, dim, theta, thres)
    tmp_thres = thres(dim, theta);
    count = 0;
    for i=1:100
        if train_y(i) ~= s*sign(train_x(i,dim)-tmp_thres)
            count = count + 1;
        end
    end
    error = count/100;
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