close all;
nSamples=10000;
seed=2;
rand('seed',seed);
NSurvive=0;
rateSurvive=zeros(nSamples,1);

for n=1:nSamples
    xDist=0;
    for i=1:6
        r=rand(1,1);
        direction=2*pi*r;
        xDist=xDist+cos(direction);
        if(xDist>=4)
            NSurvive=NSurvive+1;
            break;
        end
    end
    rateSurvive(n)=100*NSurvive/n;
end

plot(rateSurvive, 'r-', 'linewidth', 2);
title(sprintf('Rate of neutrons survive (seed=%d)',seed));
xlabel('Number of neutrons, n');
ylabel('Rate of neutrons survive (%)');
axis([1000,10000 0 10]);

grid on

pause
print -dpsc2 volumn.eps
close