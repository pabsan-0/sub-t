%% data loading and splitting
a = dlmread('data1.txt');
x   = a(:,2);
z   = a(:,3);
yaw = a(:,4);

x_train   = x(1:1000)
z_train   = z(1:1000)
yaw_train = yaw(1:1000)

x_valid   = x(1001:end)
z_valid   = z(1001:end)
yaw_valid = yaw(1001:end)


%% Overview of all data

figure('Position', [10 10 1300 600]);clf; hold on
subplot(3,5,[1 2]); plot(x);     legend('x','train/test split');   line([1e3 1e3],[-500 500]); ylim([-250 -205])
subplot(3,5,[6 7]); plot(z);     legend('z','train/test split');   line([1e3 1e3],[-500 500]); ylim([360 420])
subplot(3,5,[11 12]); plot(yaw); legend('yaw','train/test split'); line([1e3 1e3],[-500 500]); ylim([17 22])   
subplot(3,5,[3 8 13]); hist(x, 100);    legend('x')
subplot(3,5,[4 9 14]); hist(z, 100);	legend('z')
subplot(3,5,[5 10 15]); hist(yaw, 100); legend('yaw')


%% Demo of fitting a gaussian 
avg = mean(x_train);
variance = var(x_train);
sample = avg + randn(1, 10000)*variance;

figure()
subplot(2,1,1); hist(sample,100)
subplot(2,1,2); hist(x_train,100)


%% System model

% ts = (a(:,1) - circshift(a(:,1), 1))

Ts = 0.0386

A = [1 Ts  0 0   0 0;
     0 0   0 0   0 0;
     0 0   1 Ts  0 0;
     0 0   0 0   0 0;
     0 0   0 0   1 Ts;
     0 0   0 0   0 0];

B = zeros(6,3);

C = [1 0   0 0   0 0;
     0 0   1 0   0 0;
     0 0   0 0   1 0];

Rw= [1 0   0 0   0 0;
     0 1   0 0   0 0;
     0 0   1 1   0 0;
     0 0   0 1   0 0;
     0 0   0 0   1 0;
     0 0   0 0   0 1];
 


u = zeros(3, length(x_train))

y = [x_train, z_train, yaw_train]'

% Rv = [var(x_train)^2   0               0;
%      0              var(y_train)^2     0;
%      0              0               var(yaw_train)^2]
 
Rv = cov(y')

%% Luenberg predictor
P = [0.55 0.41 0.4 0.65 0.5 0.45]
yhat_luenberg = pablo_luenberg(u, y, P, A, B, C);

figure(); clf; hold on
plot(yhat_luenberg(3,:))
plot(yaw_train)
legend('predicted x_train', 'real x_train')


%% Kalman filter
LL = eye(6)
[yhat_kalman_tv, converging_L, converging_P] = pablo_kalman_tv(u, y, A, B, C, LL, Rw*1e-3, Rv);

figure(); clf; hold on
plot(yhat_kalman_tv(3,:))
plot(yaw_train)
legend('predicted x_train', 'real x_train')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function yhat_luenberg = pablo_luenberg(u, y, P, A, B, C);
    %%% Gets a one-step prediction from Luembergs observer for the system 
    %%% output yhat. Function inputs are:
    %%%     u: known input
    %%%     y: known output
    %%%     P: observer poles
    %%%     A, B, C: propagation & measurement equation matrices*
 
    L = (place(A',C',P))';
    
    xhat = zeros(6,max(size(u)));
    yhat = zeros(3,max(size(u)));
    
    for k = 1:max(size(u))
        xhat(:,k+1) = A * xhat(:,k) + B*u(:,k) + L*(y(:,k) - C*xhat(:,k));
        yhat(:,k) = C * xhat(:,k);
    end
    
    xhat_luenberg = xhat;
    yhat_luenberg = yhat;
end



function [yhat_kalman_tv, converging_L, converging_P] ...
            = pablo_kalman_tv(u, y, A, B, C, LL, Q, R);
    %%% Gets the output from Kalman's filter for the system 
    %%% output yhat. Also exports stationary & hopefully convergent values 
    %%% for L and P. Function inputs are:
    %%%     u: known input
    %%%     y: known output
    %%%     A, B, C: propagation & measurement equation matrices*
    %%%     LL: coefficient matrix for the process noise w
    %%%     Q, R: covariance matrices for process and measurement noise

    xhat = zeros(6,max(size(u)));
    yhat = zeros(3,max(size(u)));

    L = zeros(6,3,max(size(u)));  % filter gain
    P = zeros(6,6,max(size(u)));  % covariance estimation error
    
    % we are at instant K (known state) aiming to predict k+1
    for k = 1:max(size(u))-1
        % Propagation loop
        xhat(:,k+1) = A*xhat(:,k) + B*u(:,k);
        P(:,:,k+1) = A*P(k)*A' + LL*Q*LL';
                  
        % Upgrade loop - Theoretically this would happen at the beggining 
        % of the next loop, therefore all (k+1)'s are actually (k)'s so 
        % causality holds.
        L(:,:,k+1) = P(:,:,k+1) * C' * (C*P(:,:,k+1)*C' + R)^(-1);
        xhat(:,k+1) = xhat(:,k+1) + L(:,:,k+1)*(y(k+1) - C*xhat(:,k+1));
        P(:,:,k+1) = P(:,:,k+1) - L(:,:,k+1)*C*P(:,:,k+1);
        
        % Predict output. Extract here for std kalman filtering
        yhat(:,k+1) = C*xhat(:,k+1);
    end
    
    xhat_kalman_tv = xhat;
    yhat_kalman_tv = yhat;
    
    converging_L = L(:,:,900);
    converging_P = P(:,:,900);
end








