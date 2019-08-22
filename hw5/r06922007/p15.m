data = readInput('features.train',0);
inst = data(:,2:3);
label = data(:,1);
dataTest = readInput('features.test',0);
instTest = dataTest(:,2:3);
labelTest = dataTest(:,1);
Gamma = [0,1,2,3,4];
Eout = [];
%svm
for i=1:5
    model = svmtrain(label, inst, sprintf('-t 2 -g %f -c 0.1',10^Gamma(i)));
    [predict_label, accuracy, dec_values] = svmpredict(labelTest, instTest, model);
    Eout = [Eout, (100-accuracy(1))/100];
end
%plot a figure
plot(Gamma,Eout,'b-o');
xlabel('log10Gamma');
ylabel('Eout');

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