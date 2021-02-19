close all;
t = linspace(0,pi/100,1000);
Omega = 200;
theta = Omega*t;
lambda  = 3e8/24.125e9;
k = 2*pi/lambda;
a = 0.25;
BW1 = 2*2*a*Omega/lambda
x = (0:1:length(t)-1) ; % -> range
x=(x-(x(end)/2))*BW1/(2*300);
n = (-800:1:800)';
A = sum(besselj(n,2*k*a).*1i.^n.*exp(1i*n*theta),1);
S = fftshift(fft(A));
figure
plot(x,abs(S));
BW2 = 550*Omega/(2*pi)



