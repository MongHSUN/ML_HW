round = 1;
file = fopen('hw2_train.dat','r');
formatSpec = '%f %f %f %f %f %f %f %f %f %f';
size = [10 Inf];
data = fscanf(file, formatSpec, size);
data = data';
fclose(file);
min = 100; theta = 0; s = 0; dimension = 0;
for i=1:9
    %enumerate all possible threshold
    x = data(:,i);
    y = data(:,10);
    sort_x = sort(x);
    parameter = zeros(198,2); %theta, s
    for j=1:99
        parameter(2*j-1,1) = mean([sort_x(j),sort_x(j+1)]);
        parameter(2*j-1,2) = 1;
        parameter(2*j,1) = mean([sort_x(j),sort_x(j+1)]);
        parameter(2*j,2) = -1;
    end
    %decision stump
    for j=1:198
        count = 0;
        for k=1:100
            if y(k)~=parameter(j,2)*sign(x(k)-parameter(j,1))
                count = count + 1;
            end
        end
        if count<min || (count==min &&rand()<0.5)
            min = count;
            theta = parameter(j,1);
            s = parameter(j,2);
            dimension = i;
        end
    end
end
file = fopen('hw2_test.dat','r');
testing = fscanf(file, formatSpec, size);
testing = testing';
fclose(file);
x = testing(:,dimension);
y = testing(:,10);
count = 0;
for i=1:1000
    if y(i)~=s*sign(x(i)-theta)
        count = count + 1;
    end
end
Ein = min/100;
Eout = count/1000;
fprintf('Ein = %f\n',mean(Ein));
fprintf('Eout = %f\n',mean(Eout));