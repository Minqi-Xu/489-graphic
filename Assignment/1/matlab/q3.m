close all;
nSamples=10000;
seed=758;
rand('seed',seed);
Ninside=0;
Volumn=zeros(nSamples,1);

for m=1:nSamples
    x=rand(1,1);
    y=rand(1,1);
    z=rand(1,1);
    if(((x^2+sin(y))<=z) & ((x-z+exp(y))<=1))
        Ninside=Ninside+1;
    end
    Volumn(m)=Ninside/m;
end

plot(Volumn,'r-','linewidth',2);
title(sprintf('Volumn Approximations (seed=%d)',seed));
xlabel('Number of Smaples');
ylabel('Estimated Volumn');
axis([1000,10000 0.1 0.2]);
grid on

pause
print -dpsc2 volumn.eps
close