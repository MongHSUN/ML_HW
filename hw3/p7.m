n = 1000;
t = 1000;
Eout = [];
for i=1:t
    train = dataGenerate();
    train = reshape(train,7,1000)';
    test = dataGenerate();
    test = reshape(test,7,1000)';
    x = train(:,1:6);
    y = train(:,7);
    inverse = pinv(x); %pseudo inverse
    w = inverse*y;
    Eout = [Eout, error(test,w)];
    disp(i);
end
fprintf('aver Eout : %f\n',sum(Eout)/t);
figure(1);
hist(Eout,1000*(max(Eout)-min(Eout)));

function output = dataGenerate
    output = [];
    data = -1 + (1+1)*rand(1,2000);
    data = reshape(data,2,1000)';
    noise = rand(1,1000);
    for i=1:1000
        x1 = data(i,1);
        x2 = data(i,2);
        if noise(i)<0.1 %10% flip
            output = [output, 1, x1, x2, x1*x2, x1^2, x2^2, -sign(x1^2+x2^2-0.6)];
        else
            output = [output, 1, x1, x2, x1*x2, x1^2, x2^2, sign(x1^2+x2^2-0.6)];
        end
    end
end

function output = error(data, w)
    output = 0;
    [m, n] = size(data);
    for i=1:m
        y = data(i,n);
        x = data(i,1:n-1);
        if sign(dot(x,w))~=y
            output = output + 1;
        end
    end
    output = output/m;
end