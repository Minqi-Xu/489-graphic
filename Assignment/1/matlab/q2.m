close all;
nSample=10000;
seed=758;
rand('seed',seed);
Nsatisfied=0;
proportion=0;

for m=1:nSample
    d=rand(1,1);
    if((40*(d^2)+7)>(43*d))
        Nsatisfied=Nsatisfied+1;
    end
    if(m==1000)
        proportion=Nsatisfied/m;
        sprintf('Number of points = %d, proportion satisfied = %f',m,proportion)
    end
end
proportion=Nsatisfied/10000;
sprintf('Number of points = 10000, proportion satisfied = %f',proportion)
