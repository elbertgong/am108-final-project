%% Kevin Stephen
%% PCA to bring Data to Low Dimensions
%% ------------------------------------------------------------------------


%% Y is the data coming in from 3-bit flip flop example
Y=csvread('no_inputs.csv');
Y = Y';
[M,N]=size(Y);
Nplot=200;
time=1:0.3:3000;

%% plot data:
figure(1); clf
set(0,'defaulttextfontsize',18); set(0,'defaultaxesfontsize',18);

for i=1:N
    hlY = plot(time(1,1:N), Y(i,1:N),'-b')
    hold on
end
hold on
xlabel('time')
ylabel('vector component')
title('Trajectories vs Time')

disp('Enter to continue')
pause

fprintf('\n\n---------------------------------------------------------------\n')
fprintf(1,['First approach: analyze relations between the two data \n' ...
           'sets using principal component analysis.\n'])
fprintf(1,[' That is, form a single vector of all data, ' ...
           'calculate multivariate EOFs:\n'])
fprintf(1,['(requires normalizing data to have similar units, e.g., ' ...
            'by standard deviation)\n']) 
fprintf('---------------------------------------------------------------\n')
X=Y;
[XM,XN]=size(X);
[XU,XS,XV]=svd(X);
disp('perform svd analysis of data matrix, [XU,XS,XV]=svd(X):')
disp(['singular values diag(XS) of data matrix indicate that first three modes '...
      'are most important:'])
disp(diag(XS)')
%disp('The corresponding three PCs are:')
%disp(XU(:,1:3));



