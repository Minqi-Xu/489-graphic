close all;
nSamples=2000;
seed=758;
rand('seed',seed);
Ninside=0;
rateInside=zeros(nSamples,1);

for n=1:nSamples
    d=rand(1,1);
    x=4*d-2;
    d=rand(1,1);
    y=2*d-1;
    if((x^2+y^2)<=4)
        Ninside=Ninside+1;
    end
    rateInside(n)=Ninside/n
end

plot(rateInside,'r-','linewidth',2);
title(sprintf('Rate of points inside ellipse (seed=%d)',seed));
xlabel('Number of Points ploted');
ylabel('Rate of Points inside ellipse');
grid on

pause

print -dpsc2 dart.eps
close

