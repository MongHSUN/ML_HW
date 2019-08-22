%read train & test data
formatSpec = '%f %f %f';
sizeSpec = [3 Inf];
file1 = fopen('hw4_train.dat','r');
train = fscanf(file1, formatSpec, sizeSpec);
train = train';
fclose(file1);
file2 = fopen('hw4_test.dat','r');
test = fscanf(file2, formatSpec, sizeSpec);
test = test';
fclose(file2);
%ridge regression
lambda = -10:2;
Ein = []; Eout = [];
for i=1:13
    x = matrixGenerate(train);
    y = train(:,3);
    w = inv(x'*x+10^lambda(i)*eye(3))*x'*y;
    Ein = [Ein, error(train,w)];
    Eout = [Eout, error(test,w)];
end
%plot a figure
plot(lambda,Ein,'b');
hold on;
plot(lambda,Eout,'r');
xlabel('log10lambda');
ylabel('Error');
legend('Ein','Eout');
    
function output = matrixGenerate(data)
    output = [];
    [m, n] = size(data);
    for i=1:m
        output = [output, 1, data(i,1:n-1)];
    end
    output = reshape(output,n,m)';
end

function output = error(data, w)
    output = 0;
    [m, n] = size(data);
    for i=1:m
        y = data(i,n);
        x = [1,data(i,1:n-1)];
        if sign(dot(x,w))~=y
            output = output + 1;
        end
    end
    output = output/m;
end