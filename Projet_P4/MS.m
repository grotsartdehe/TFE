%% Initialisation
close all;
clear all;

%% Constantes
c = 3e8;
f_0 = 24e9;
Delta23 = 0.0225;
Delta12 = 0.036;
k = 2*pi*f_0/c;
Number = 119;
j = sqrt(-1) ;
Ns = 1000 ;
w0 = 2*pi*f_0 ;
folder = '/Users/Tanguy/Documents/UNIVERSITE/Q6/LFSAB1508 - Projet P4 Elec/MS/donneeroute';

%% MS's
reply = input('Quel MS voulez-vous afficher? (Uniquement le chiffre du MS, x pour tous les afficher)   ','s');
%% MS0
if strcmp(reply,'0') | strcmp(reply,'x')
    load(fullfile(folder,'dataGroup10.mat'));
    signal = zeros(1,200597) ;
    i=[1:200597];
    signal = sig_.*exp(j*w0*i);
    transformee = fft(real(signal),Ns)/Ns ;
    X = fftshift(transformee) ;
    freqHz = fs_*(-((Ns-1)/2):((Ns-1)/2))/Ns ;
    figure;
    stem(freqHz,abs(X));
    title('FFT du signal mesuré (sig\_)')
    xlabel('f [Hz]');
    ylabel('|Y| [dB]');
    grid on;
    % Cas 1
    f0 = 10000 ;
    fs = 50000 ;
    Ts = 1/fs ;
    Ns = 100;
    Ns2 = Ns+1 ;
    Ns3 = Ns*10 ;
    n = 0:Ns-1 ;
    n2 = 0:Ns2-1 ;
    n3 = 0:Ns3-1 ;
    x = cos(2*pi*f0*n*Ts)' ;
    x2 = cos(2*pi*f0*n2*Ts)' ;
    x3 = cos(2*pi*f0*n3*Ts)' ;
    X_beta = fft(x,Ns)/Ns ;
    X_beta2 = fft(x2,Ns2)/Ns2 ;
    X_beta3 = fft(x3,Ns3)/Ns3 ;
    X = fftshift(X_beta) ;
    Z = fftshift(X_beta2) ;
    Z2 = fftshift(X_beta3) ;
    freqHz = fs*(-((Ns-1)/2):((Ns-1)/2))/Ns;
    freqHz2 = fs*(-((Ns2-1)/2):((Ns2-1)/2))/Ns2;
    freqHz3 = fs*(-((Ns3-1)/2):((Ns3-1)/2))/Ns3;
    
    figure
    subplot(3,1,1);
    stem(freqHz,real(X));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal cos(\Omega_0 n) pour Ns = 100 ');
    subplot(3,1,2);
    stem(freqHz2,real(Z));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal cos(\Omega_0 n) pour Ns = 101 ');
    subplot(3,1,3);
    stem(freqHz3,real(Z2));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal cos(\Omega_0 n) pour Ns = 1000 ');
    
    % Cas 2
    f0 = 10000 ;
    i = sqrt(-1) ;
    Ts = 1/fs ;
    Ns = 100;
    Ns2 = Ns+1 ;
    Ns3 = Ns*10 ;
    n = 0:Ns-1 ;
    n2 = 0:Ns2-1 ;
    n3 = 0:Ns3-1 ;
    x = exp(2*pi*i*f0*n*Ts)' ;
    x2 = exp(2*pi*i*f0*n2*Ts)' ;
    x3 = exp(2*pi*i*f0*n3*Ts)' ;
    X_beta = fft(x,Ns)/Ns ;
    X_beta2 = fft(x2,Ns2)/Ns2 ;
    X_beta3 = fft(x3,Ns3)/Ns3 ;
    X = fftshift(X_beta) ;
    Z = fftshift(X_beta2) ;
    Z2 = fftshift(X_beta3) ;
    freqHz = fs*(-((Ns-1)/2):((Ns-1)/2))/Ns;
    freqHz2 = fs*(-((Ns2-1)/2):((Ns2-1)/2))/Ns2;
    freqHz3 = fs*(-((Ns3-1)/2):((Ns3-1)/2))/Ns3;
    
    figure
    subplot(3,1,1);
    stem(freqHz,real(X));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(2\piif_{0}t) pour Ns = 100 ');
    subplot(3,1,2);
    stem(freqHz2,real(Z));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(2\piif_{0}t) pour Ns = 101 ');
    subplot(3,1,3);
    stem(freqHz3,real(Z2));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(2\piif_{0}t) pour Ns = 1000 ');
    
    % Cas 3
    f0 = 10000 ;
    i = sqrt(-1) ;
    Ts = 1/fs ;
    Ns = 100;
    Ns2 = Ns+1 ;
    Ns3 = Ns*10 ;
    n = 0:Ns-1 ;
    n2 = 0:Ns2-1 ;
    n3 = 0:Ns3-1 ;
    x = exp(-2*pi*i*f0*n*Ts)' ;
    x2 = exp(-2*pi*i*f0*n2*Ts)' ;
    x3 = exp(-2*pi*i*f0*n3*Ts)' ;
    X_beta = fft(x,Ns)/Ns ;
    X_beta2 = fft(x2,Ns2)/Ns2 ;
    X_beta3 = fft(x3,Ns3)/Ns3 ;
    X = fftshift(X_beta) ;
    Z = fftshift(X_beta2) ;
    Z2 = fftshift(X_beta3) ;
    freqHz = fs*(-((Ns-1)/2):((Ns-1)/2))/Ns;
    freqHz2 = fs*(-((Ns2-1)/2):((Ns2-1)/2))/Ns2;
    freqHz3 = fs*(-((Ns3-1)/2):((Ns3-1)/2))/Ns3;
    
    figure
    subplot(3,1,1);
    stem(freqHz,real(X));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(-2\piif_{0}t) pour Ns = 100 ');
    subplot(3,1,2);
    stem(freqHz2,real(Z));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(-2\piif_{0}t) pour Ns = 101 ');
    subplot(3,1,3);
    stem(freqHz3,real(Z2));
    xlabel('Fréquence vraie [Hz]');
    ylabel('|Y| [dB]');
    title('Module de la fft du signal exp(-2\piif_{0}t) pour Ns = 1000 ');
    % Signal en plus
    load('dataGroup10.mat')
    FrequencePorteuse = 24*10^9 ;
end

%% MS1
if strcmp(reply,'1') | strcmp(reply,'x') | strcmp(reply,'3')
    y=[0 153.6];
    x=[-150 150];
    %% Calibration
    load(fullfile(folder,'Microdop1_CAL'));
    Number = 288;
    CAL=[a_1 a_2 a_3];
    
    %% Figure
    for i = 1: Number
        load(fullfile(folder,['Microdop1_' num2str(i) '.mat']));
        C = abs(fft(fftshift(fft(a_1-CAL(1)))')).^2+abs(fft(fftshift(fft(a_2-CAL(2)))')).^2+abs(fft(fftshift(fft(a_3-CAL(3)))')).^2;
        pause(0.2);
        imagesc(x,y,flipud(mag2db(C)));
        ax = gca;
        ax.YDir = 'normal';
        xlabel('Velocity [km/h]');
        ylabel('Range [m]');
        title(['Measure number ' num2str(i)]);
    end
end

%% MS2
if strcmp(reply,'2') | strcmp(reply,'x')
    rep2= input('Voulez vous afficher la représentation de la corrélation? Si non, nous afficherons l évolution des angles   ','s');
    %% Calculs des images
    if strcmp(rep2,'oui')
        u_x = linspace(-1,1,256);
        u_y = linspace(-1,1,256);
    else
        u_x=linspace(cos(pi/2-0.1744),cos(pi/2+0.1744),256);
        u_y=linspace(cos(pi/2-0.28148),cos(pi/2+0.28148),256);
    end
    
    load(fullfile(folder,'dataazimut_CAL'));
    Number = 118;
    a_1c=a_1;
    a_2c=a_2;
    a_3c=a_3;
    for i = 1:Number
        load(fullfile(folder,['dataazimut_' num2str(i) '.mat']));
        a_1 = a_1-a_1c;
        a_2 = a_2-a_2c;
        a_3 = a_3-a_3c;
        C1 = abs(fft(fftshift(fft(a_1))')).^2;
        %C2 = abs(fft(fftshift(fft(a_2))')).^2;
        %C3 = abs(fft(fftshift(fft(a_3))')).^2;
        
        r_x= [0.036/2 -0.036/2 -0.036/2];
        r_y= [-0.0225/2 -0.0225/2 0.0225/2];
        rpos = C1 == max(max(C1));
        C = abs(a_1(rpos)*exp(-1i*k*(u_x'*ones(1,length(u_x))*r_x(1)+(u_y'*ones(1,length(u_x)))'*r_y(1)))+ a_2(rpos)*exp(-1i*k*(u_x'*ones(1,length(u_x))*r_x(2)+(u_y'*ones(1,length(u_x)))'*r_y(1))) + a_3(rpos)*exp(-1i*k*(u_x'*ones(1,length(u_x))*r_x(3)+(u_y'*ones(1,length(u_x))*r_y(3))')));
        [x,y] = find(C==max(max(C)));
        u_xx=u_x(x(1));
        u_yy=u_y(y(1));
        phi(i)=rad2deg(acos(u_yy));
        theta(i)=rad2deg(acos(u_xx));
        if strcmp(rep2,'oui')
            waitforbuttonpress
            %pause(0.1);
            imagesc(real(C));%,[-4500 4500]);
            xticklabels = -1:0.2:1;
            xticks = linspace(1, 256, numel(xticklabels));
            set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
            yticklabels = -1:0.2:1;
            yticks = linspace(1, 256, numel(yticklabels));
            set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)
            colorbar;
            xlabel('u_y');
            ylabel('u_x');
            title(['Measure number ' num2str(i)]);
        end
    end
    outlierphi = isoutlier(phi);
    outliertheta = isoutlier(theta);
    if strcmp(rep2,'non')
        theta=theta-9.75;%.5697;
        phi=phi-2.19;
        %phi_f=phi_f+90-phi_f(1);
        for i=1:Number-1
            if outlierphi(i)==true
                if outlierphi(i+1)==true
                    phi(i)=(phi(i-1)+phi(i+2))/2;
                else
                    phi(i)=(phi(i-1)+phi(i+1))/2;
                end
            end
            if outliertheta(i)==true
                if outliertheta(i+1)==true && i<Number-2
                    if outliertheta(i+2)==true && i<Number-3
                        theta(i)=(theta(i-1)+theta(i+3))/2;
                    else
                        theta(i)=(theta(i-1)+theta(i+2))/2;
                    end
                else
                    theta(i)=(theta(i-1)+theta(i+1))/2;
                end
            end
            if abs(theta(i+1)-theta(i))>20
                theta(i)=0;
            end
            if phi(i+1)-phi(i)>=15 && theta(i)~=0% && theta(i+1)~=0 && theta(i-1)~=0 %&& abs(theta(i+1)-theta(i))<8 && theta(i)~=0%<8%theta(i+1)-theta(i))~=0%<8
                phi(i+1:Number)=phi(i+1:Number)-2*16;
            end
            if phi(i+1)-phi(i)<=-15 && theta(i)~=0% && theta(i+1)~=0 && theta(i-1)~=0 %&& abs(theta(i+1)-theta(i))<8 && theta(i)~=0%<8%abs(theta(i+1)-theta(i))~=0%<8
                phi(i+1:Number)=phi(i+1:Number)+2*16;
            end
            %if theta(i+1)-theta(i)>=17 && phi(i+1)>70 && phi(i+1)<115 && theta(i)~=0
            %    theta(i+1:Number)=theta(i+1:Number)-2*10;
            %end
            %if theta(i+1)-theta(i)<=-17 && phi(i+1)>70 && phi(i+1)<115 && theta(i)~=0
            %    theta(i+1:Number)=theta(i+1:Number)+2*10;
            %end
            %if phi(i) <= 70
            %    phi(i)=70;
            %end
            %if phi(i) >= 110
            %    phi(i)=110;
            %end
            %if theta(i)<=65 && theta(i)~=0
            %    theta(i)=65;
            %end
            %if theta(i)>=115
            %    theta(i)=115;
            %end
        end
        
        
        phi = spline([1:Number],phi,[1:2:Number]); % Eventuellement enlever
        theta = spline([1:Number],theta,[1:2:Number]); % Eventuellement enlever
        phi = pchip([1:2:Number],phi,[1:0.03:Number]); % Eventuellement enlever
        theta = pchip([1:2:Number],theta,[1:0.03:Number]); % Eventuellement enlever
        
        figure;
        subplot(2,1,1);
        plot([1:0.03:Number],phi)%plot([1:0.03:Number],phi)
        title('\phi évalué pour chaque mesure');
        xlabel('Measure number');
        ylabel('\phi [°]');
        grid on;
        subplot(2,1,2);
        plot([1:0.03:Number],theta)%plot([1:0.03:Number],theta)
        title('\theta évalué pour chaque mesure');
        xlabel('Measure number');
        ylabel('\theta [°]');
        grid on;
        %figure;
        %plot([1:Number],phi)%plot([1:0.03:Number],phi)
        %title('\phi évalué pour chaque mesure');
        %xlabel('Measure number');
        %ylabel('\phi [°]');
        %grid on;
    end
    
    c = 3e8;
f = 24e9;
d = 0.0225;
d2 = 0.036;
deph=linspace(-pi,pi,1000);
zeros(5,1000);
k=[-2 -1 0 1 2]';
figure;
phi=(asin(c*deph/(2*pi*f*d)+k*c/(f*d)));
phi(1,1:699)=NaN;
phi(5,303:1000)=NaN;
plot(rad2deg(deph),rad2deg(phi));
hold on;
plot(rad2deg(deph),rad2deg(ones(1,length(deph))*phi(1,length(deph))),'k');
hold on;
plot(rad2deg(deph),rad2deg(ones(1,length(deph))*phi(2,length(deph))),'k');
hold on;
plot(rad2deg(deph),rad2deg(ones(1,length(deph))*phi(3,length(deph))),'k');
hold on;
plot(rad2deg(deph),rad2deg(ones(1,length(deph))*phi(4,length(deph))),'k');
hold on;
plot(rad2deg(deph),rad2deg(ones(1,length(deph))*phi(5,length(deph))),'k');
title('\phi en fonction du déphasage')
xlabel('Déphasage [°]');
ylabel('\phi [°]');
legend('k=-2','k=-1','k=0','k=1','k=2')
grid on;

    
    
end

%% MS3
if strcmp(reply,'3') | strcmp(reply,'x')
    k = 2*pi*f_0/c;
    a=0.05;
    n=[-230:230];
    C = j.^n.*besselj(n,2*k*a)
    figure;
    plot(n,abs(C),'-')
    title('Spectre micro-doppler')
    xlabel('n')
    ylabel('C_n')
    hold on;
end

%% Fichier de mesures disponibles :

% load(fullfile(folder,'firstdataroad_CAL'));
% Number = 119;
% load(fullfile(folder,['firstdataroad_' num2str(i) '.mat']));

% load(fullfile(folder,'dataazimut_CAL'));
% Number = 118;
% load(fullfile(folder,['dataazimut_' num2str(i) '.mat']));

% load(fullfile(folder,'datatheta_CAL'));
% Number = 52;
% load(fullfile(folder,['datatheta_' num2str(i) '.mat']));

% load(fullfile(folder,'dataspirale_CAL'));
% Number = 92;
% load(fullfile(folder,['dataspirale_' num2str(i) '.mat']));

% load(fullfile(folder,'dataphi2_CAL'));
% Number = 56;
% load(fullfile(folder,['dataphi2_' num2str(i) '.mat']));

% load(fullfile(folder,'dataphi_CAL'));
% Number = 56;
% load(fullfile(folder,['dataphi_' num2str(i) '.mat'])); PAS SUR DE LA
% CALIBRATION!!

% load(fullfile(folder,'Microdop1_CAL'));
% Number = 288;
% load(fullfile(folder,['Microdop1_' num2str(i) '.mat']));

% load(fullfile(folder,'theta0phi0_CAL'));
% Number = 33;
% load(fullfile(folder,['theta0phi0_' num2str(i) '.mat']));