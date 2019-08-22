%read train & test data
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f';
sizeSpec = [21 Inf];
file1 = fopen('hw3_train.dat','r');
train = fscanf(file1, formatSpec, sizeSpec);
train = train';
fclose(file1);
file2 = fopen('hw3_test.dat','r');
test = fscanf(file2, formatSpec, sizeSpec);
test = test';
fclose(file2);
%logistic regression for era = 0.001
Eout = [];
EoutSto = [];
w = zeros(1,21);
wSto = zeros(1,21);
t = 2000;
eta = 0.001;
index = 1;
for i=1:t
    gradient = gradientDescent(train,w);
    w = w + eta * gradient;
    Eout = [Eout, error(test,w)];
    stoGradient = stochasticGradient(train, wSto, index);
    wSto = wSto + eta * stoGradient;
    EoutSto = [EoutSto, error(test,wSto)];
    index = index + 1;
    if index > 1000
        index = 1;
    end
    disp(i);
end
%logistic regression for era = 0.01
Eout2 = [];
EoutSto2 = [];
w = zeros(1,21);
wSto = zeros(1,21);
eta = 0.01;
for i=1:t
    gradient = gradientDescent(train,w);
    w = w + eta * gradient;
    Eout2 = [Eout2, error(test,w)];
    stoGradient = stochasticGradient(train, wSto, index);
    wSto = wSto + eta * stoGradient;
    EoutSto2 = [EoutSto2, error(test,wSto)];
    index = index + 1;
    if index > 1000
        index = 1;
    end
    disp(i);
end
fprintf('Eout for gradientDescent with eta = 0.001 : %f\n',Eout(t));
fprintf('Eout for stochasticGradient with eta = 0.001 : %f\n',EoutSto(t));
fprintf('Eout for gradientDescent with eta = 0.01 : %f\n',Eout2(t));
fprintf('Eout for stochasticGradient with eta = 0.01 : %f\n',EoutSto2(t));
t = [1:2000];
figure(1);
plot(t,Eout,'o',t,EoutSto,'r');
xlabel('t');
ylabel('Eout');
figure(2);
plot(t,Eout2,'o',t,EoutSto2,'r');
xlabel('t');
ylabel('Eout');

function output = gradientDescent(train, w)
    output = zeros(1,21);
    for i=1:1000
        y = train(i,21);
        x = train(i,1:20);
        x = [1,x];
        sigmoid = 1/(1+exp(y*dot(w,x)));
        output = output + sigmoid * y * x;
    end
    output = output/1000;
end

function output = stochasticGradient(train, w, index)
    y = train(index,21);
    x = train(index,1:20);
    x = [1,x];
    sigmoid = 1/(1+exp(y*dot(w,x)));
    output = sigmoid * y * x;
end

function output = error(data, w)
    output = 0;
    range = size(data);
    for i=1:range(1)
        y = data(i,21);
        x = data(i,1:20);
        x = [1,x];
        if sign(dot(x,w))~=y
            output = output + 1;
        end
    end
    output = output/range(1);
end