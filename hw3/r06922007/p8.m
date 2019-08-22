%read train & test data
formatSpec = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f';
sizeSpec = [21 Inf];
file1 = fopen('hw3_train.dat','r');
train = fscanf(file1, formatSpec, sizeSpec);
train = train';
fclose(file1);
%logistic regression for era = 0.001
Ein = [];
EinSto = [];
w = zeros(1,21);
wSto = zeros(1,21);
t = 2000;
eta = 0.001;
index = 1;
for i=1:t
    gradient = gradientDescent(train,w);
    w = w + eta * gradient;
    Ein = [Ein, error(train,w)];
    stoGradient = stochasticGradient(train, wSto, index);
    wSto = wSto + eta * stoGradient;
    EinSto = [EinSto, error(train,wSto)];
    index = index + 1;
    if index > 1000
        index = 1;
    end
    disp(i);
end
%logistic regression for era = 0.01
Ein2 = [];
EinSto2 = [];
w = zeros(1,21);
wSto = zeros(1,21);
eta = 0.01;
for i=1:t
    gradient = gradientDescent(train,w);
    w = w + eta * gradient;
    Ein2 = [Ein2, error(train,w)];
    stoGradient = stochasticGradient(train, wSto, index);
    wSto = wSto + eta * stoGradient;
    EinSto2 = [EinSto2, error(train,wSto)];
    index = index + 1;
    if index > 1000
        index = 1;
    end
    disp(i);
end
fprintf('Ein for gradientDescent with eta = 0.001 : %f\n',Ein(t));
fprintf('Ein for stochasticGradient with eta = 0.001 : %f\n',EinSto(t));
fprintf('Ein for gradientDescent with eta = 0.01 : %f\n',Ein2(t));
fprintf('Ein for stochasticGradient with eta = 0.01 : %f\n',EinSto2(t));
t = [1:2000];
figure(1);
plot(t,Ein,'o',t,EinSto,'r');
xlabel('t');
ylabel('Ein');
figure(2);
plot(t,Ein2,'o',t,EinSto2,'r');
xlabel('t');
ylabel('Ein');

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