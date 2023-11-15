close all;
nSamples=100000;
seed=758;
rand('seed',seed);
count=0;  % count stores the number of satisfied samples
Pvalue=zeros(nSamples,1);  % Pvalue stores the probabilities wrt number of samples

% Set the coordinate on the board, and assume that the two parallel lines
%   are y=0 and y=1. Since the board are infinitely on x and y direction,
%   therefore, when make a random point, x-value for the point is not 
%   important, we only need to randomize the y-value of the point and the
%   direction of the stick.
for m=1:nSamples
    % we first random select the y-value of the center of the stick
    y=rand(1,1);
    % then we get the direction of the stick
    r=rand(1,1);
    % Due to the symmetry, we only need to random the direction in [0,pi)
    d=pi*r;
    % since the stick is unit long, only y(center)=0.5 and d=pi/2 can
    % intersects both parallel lines. We can separate the problem into 3
    % cases.

    % first case is y==0.5
    % this case, stick has no way to intersects one of the lines, thus
    % nothing changes.

    if(y>0.5)
        % in this case, stick can only intersects with the upper line, or
        % does not intersect with any lines.
        % len is the distance from center to the nearest line
        len = 1-y;
        % angle is the angle between the stick and the norm of the line
        angle = abs(pi/2 - d);
        if((0.5*cos(angle))>=len)
            % in this case stick intersects with the upper line
            count=count+1;
        end
    elseif(y<0.5)
        % similar to the previous case, but this time is lower line.
        % len is the distance from center to the nearest line
        len = y;
        % angle is the angle between the stick and the norm of the line
        angle = (pi/2 - d);
        if((0.5*cos(angle))>=len)
            % in this case stick intersects with the lower line
            count=count+1;
        end
    end
    % Pvalue stores the percentage, so 100 is multiplied
    Pvalue(m)=(count/m)*100;
end

plot(Pvalue,'r-','linewidth',2);
title(sprintf('Probability stick intersects one of the lines (seed=%d)',seed));
xlabel('Number of Samples');
ylabel('Probability (%)');
axis([1000,100000,60,70]);
grid on

pause
print -dpsc2 prob.eps
close
