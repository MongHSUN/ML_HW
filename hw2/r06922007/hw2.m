round = 1000;
Ein = [];
Eout = []; 
for i=1:round
    %data generation
    x = -1 + (1+1)*rand(1,20);
    noise = rand(1,20);
    y = [];
    for j=1:20
        if noise(j)<0.2 %20% flip
            y = [y, -sign(x(j))];
        else
            y = [y, sign(x(j))];
        end
    end
    %enumerate all possible threshold
    sort_x = sort(x);
    parameter = zeros(40,2); %theta, s
    for j=1:20
        if j~=20
            parameter(2*j-1,1) = mean([sort_x(j),sort_x(j+1)]);
            parameter(2*j-1,2) = 1;
            parameter(2*j,1) = mean([sort_x(j),sort_x(j+1)]);
            parameter(2*j,2) = -1;
        else
            parameter(2*j-1,1) = mean([1,sort_x(j)]);
            parameter(2*j-1,2) = 1;
            parameter(2*j,1) = mean([1,sort_x(j)]);
            parameter(2*j,2) = -1;
        end
    end
    %decision stump
    min = 20; theta = []; s = [];
    for j=1:40
        count = 0;
        for k=1:20
            if y(k)~=parameter(j,2)*sign(x(k)-parameter(j,1))
                count = count + 1;
            end
        end
        if count<min
            min = count;
            theta = parameter(j,1);
            s = parameter(j,2);
        elseif count==min
            theta = [theta, parameter(j,1)];
            s = [s, parameter(j,2)];
        end
    end
    sizeTies = size(theta);
    index = randi(sizeTies(2)); %randomly choose to break ties
    Ein = [Ein, min/20];
    Eout = [Eout, (0.5+0.3*s(index)*(abs(theta(index))-1))];
end
p= plot(Ein,Eout,'ko');
xlabel('Ein');
ylabel('Eout');
grid on;
fprintf('Ein = %f\n',mean(Ein));
fprintf('Eout = %f\n',mean(Eout));