function [Max,  FINALE] = MS1_code()
close all;
load('calibration.mat');
a_1_cal = a_1;
a_2_cal = a_2;
a_3_cal = a_3;
clear a_1 a_2 a_3;
F = dir('*.mat');

X_cal1 = fft2(a_1_cal');
X_cal2 = fft2(a_2_cal');
X_cal3 = fft2(a_3_cal');
AMP = zeros(length(F),3);
for ii = 2:length(F)
    fid = fopen(F(ii).name);
    load(F(ii).name);
    X1 = fft2(a_1');
    X2 = fft2(a_2');
    X3 = fft2(a_3');
    T1 = ((flip(fftshift((X_cal1-X1),2),1)).^2);
    T2 = ((flip(fftshift((X_cal2-X2),2),1)).^2);
    T3=  ((flip(fftshift((X_cal3-X3),2),1)).^2);
    T1bis = abs(T1);
    T2bis = abs(T2);
    T3bis = abs(T3);
    
     A1 = max(max(T1bis));
    [x1,y1]=find(T1bis==A1); I1T = 256*(y1-1)+x1;
     A2= max(max(T2bis));
    [x2,y2]=find(T2bis==A2); I2T = 256*(y2-1)+x2;
    A3= max(max(T3bis));
    [x3,y3]=find(T3bis==A3); I3T = 256*(y3-1)+x3;
    A = [A1 A2 A3];
    IFbis = [I1T I2T I3T];
    [AF,IFT] = max(A);
    IF = IFbis(IFT);
    AMP(ii,:) = [T1(IF) T2(IF) T3(IF)];  
    fclose(fid);
end
lambda  = 3e8/24.125e9;
lambda = 2*pi/lambda;
Ux = linspace(-1,1,length(F));
Uy = linspace(-1,1,length(F));
Q = zeros(length(F)^2,length(F));

for k=1:length(F)
    for i=1:length(F)
        for l=1:length(F)     
                    Q(length(F)*(l-1)+i,k) = AMP(k,1).*exp(-1i*lambda*(Uy(l)*36e-3))+AMP(k,2)+AMP(k,3).*exp(-1i*lambda*(Ux(i)*22.5e-3));
        end
    end
end
% [Max, FINALE]= max(real(Q));
% IndexJ = fix(FINALE/93)+1;
% IndexI = mod(FINALE,93);
% for h=1:93
% if IndexI(h) == 0 
%     IndexJ(h)=IndexJ(h)-1;
%     IndexI(h)=93;
% end
% end
% DiX = Ux(IndexI); 
% DiY = Uy(IndexJ);
%Q = reshape(Q(:,25),[93 93]);
ploter(Q,length(F));
end

function ploter(F,L)
Ux = linspace(-1,1,L);
Uy = linspace(-1,1,L);

figure
count = 0;
for ii = 1:L
    imagesc(Ux,Uy,real(reshape(F(:,ii),[L L])));
    M = getframe(gcf);
    pause(0.3);
    count = count +1
end
end


