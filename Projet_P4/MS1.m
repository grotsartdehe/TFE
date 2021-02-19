load('calibration.mat');
a_1_cal = a_1;
a_2_cal = a_2;
a_3_cal = a_3;
clear a_1 a_2 a_3;
f_s = 3.413e6;
f_0=24e9;
N_s=256;
f_r=22.1 ;
c = 3e8;
F = dir('*.mat');
figure;
X_cal1 = fft2(a_1_cal');
w_0 = 2*pi*24e9;
BW = 250e6;
f_n1= (0:1:length(X_cal1)-1)*(c*256/(2*BW*256)) ; % -> range
f_n2= (0:1:length(X_cal1)-1)*(c*pi*f_s*3.6*2/(2*w_0*N_s*256));% -> doppler, speed
f_n2=(f_n2-(f_n2(end)/2));

% writerObj = VideoWriter('MS1.avi');
% writerObj.FrameRate = 10;
% writerObj.Quality = 95;
% open(writerObj);
for ii = 2:length(F)
    fid = fopen(F(ii).name);
    load(F(ii).name);
    X1 = fft2(a_1');
    T = log(abs(flip(fftshift((X_cal1-X1),2),1)).^2);
    imagesc(flip(f_n2),f_n1,T);set(gca,'YDir','normal');
    colorbar;
    hold on;
    M = getframe(gcf);
%     writeVideo(writerObj,M);
    fclose(fid);
end


% close(writerObj);
clear all;