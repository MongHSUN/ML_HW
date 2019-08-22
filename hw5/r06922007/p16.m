data = readInput('features.train',0);
Gamma = [-1,0,1,2,3];
count = [0,0,0,0,0];
%svm
for i=1:100
    index = randperm(size(data,1));
    data = data(index,:);
    inst = data(1001:7291,2:3);
    label = data(1001:7291,1);
    instVal = data(1:1000,2:3);
    labelVal = data(1:1000,1);
    Eval = 0;
    id = -1;
    for j=1:5
        model = svmtrain(label, inst, sprintf('-t 2 -g %f -c 0.1',10^Gamma(j)));
        [predict_label, accuracy, dec_values] = svmpredict(labelVal, instVal, model);
        if accuracy(1)>Eval
            Eval = accuracy(1);
            id = j;
        end
    end
    count(id) = count(id) + 1;
    disp(i);
end
%plot a figure
bar(Gamma,count);
xlabel('log10Gamma');
ylabel('count');

function data = readInput(fileName, targetDigit)
    formatSpec = '%f %f %f';
    sizeSpec = [3 Inf];
    file = fopen(fileName,'r');
    data = fscanf(file, formatSpec, sizeSpec);
    data = data';
    [m, n] = size(data);
    for i=1:m
        if data(i,1)==targetDigit
            data(i,1) = 1;
        else
            data(i,1) = -1;
        end    
    end
    fclose(file);
end