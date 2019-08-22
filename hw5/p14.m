data = readInput('features.train',0);
inst = data(:,2:3);
label = data(:,1);
C = [-3,-2,-1,0,1];
dis = [];
%svm
for i=1:5
    model = svmtrain(label, inst, sprintf('-t 2 -g 80 -c %f',10^C(i)));
    w = full(model.SVs)' * model.sv_coef;
    dis = [dis,1/norm(w)];
end
%plot a figure
plot(C,dis,'b-o');
xlabel('log10C');
ylabel('Distance');

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