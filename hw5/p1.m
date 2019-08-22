x = [1,0; 0,1; 0,-1; -1,0; 0,2; 0,-2; -2,0];
z = transform(x);
label = [-1;-1;-1;1;1;1;1];
model = svmtrain(label,x,'-t 1 -g 2 -r 1 -d 2 -c 1000000');

function z = transform(x)
    z = [];
    for i=1:7
        tmp = [0,0];
        tmp(1) = (x(i,2)^2) - 2*x(i,1) + 3;
        tmp(2) = x(i,1)^2 - 2*x(i,2) - 3;
        z = [z;tmp];
    end
end