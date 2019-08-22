data = readInput('features.train',8);
inst = data(:,2:3);
label = data(:,1);
C = [-5,-3,-1,1,3];
numSV = [];
%svm
for i=1:5
    model = svmtrain(label, inst, sprintf('-t 1 -g 1 -r 1 -d 2 -c %f',10^C(i)));
    numSV = [numSV, model.totalSV];
end
%plot a figure
plot(C,numSV,'b-o');
xlabel('log10C');
ylabel('# of SV');

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