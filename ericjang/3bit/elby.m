clear; close all;
apply_settings;
train_fname = sprintf('data/%s_train.mat',prefix);
test_fname = sprintf('data/%s_test.mat',prefix);
TrainDat = load(train_fname); % net
TestDat = load(test_fname);

fps_fname = sprintf('data/FPs_%s',prefix);
qoptim = load(fps_fname);

% transform new 1000-D points (trajectory) to new space
X = TestDat.outData.X';
X = bsxfun(@minus,X,qoptim.mu);
X = bsxfun(@rdivide,X,qoptim.we);
Y = X*qoptim.W; % rows = observations, columns = coordinates

csvwrite('erictour.csv',Y(:,1:3))
csvwrite('ericinputs.csv',TestDat.test_input')