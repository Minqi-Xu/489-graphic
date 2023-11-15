close all;
nSamples=100000;
seed = 1;
rand('seed',seed);
FormFac=zeros(nSamples,1);
sum=0;

for n=1:nSamples
    % when sampling points in the right triangle, we can simply sample in
    % the rectangle, for those points out of triangle, we can map it in to
    % the corresponding points in the triangle due to the centrosymmetric
    % to guarantee the sampling is uniform.
    r1=rand(1,1);
    x=2*r1;
    r2=rand(1,1);
    y=r2*(-2)/sqrt(3);
    if(y<(x*(-1)/sqrt(3)))
        % this case the point is out of triangle
        x=2-x;
        y=((-2)/sqrt(3))-y;
    end
    % due to parallel, the two angles in the form factor formula should be
    % the same, for simplicity, we will calculate the cos of it by
    % definition, which is b divided by the distance between random sampled point to
    % origin. Also, visibility term is 1 due to no obstacle.
    dist=sqrt(x^2+y^2+3^2);
    cosepsilon=3/dist;
    sum=sum+(cosepsilon^2/(pi*(dist^2)));
    FormFac(n)=(sum/n)*(2/sqrt(3));
end

plot(FormFac, 'r-', 'linewidth', 2);
title(sprintf('Form Factor (seed=%d)',seed));
xlabel('Number of points sampled, n');
ylabel('Form Factor');
axis([1000,100000 0 0.04]);

grid on

pause
print -dpsc2 FormFac.eps
close
