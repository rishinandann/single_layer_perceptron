clc;
clear;
close all;

ndata = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
rng(1,'twister');
classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), ...
randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);

targetsA = ones(1,ndata);
targetsB = ones(1,ndata)*-1;
dataA = [classA; targetsA];
dataB = [classB; targetsB];

%Shuffle both classes separately 
dataA_shuffled = dataA(:,randperm(ndata));
dataB_shuffled = dataB(:,randperm(ndata));

%dataset = horzcat(classA,classB);
%scatterplot(dataset);

A1_index = find(dataA_shuffled(1,:) < 0);
subsetA1 = dataA_shuffled(:,A1_index);
A2_index = find(dataA_shuffled(1,:) > 0);
subsetA2 = dataA_shuffled(:,A2_index);
subset_lengthA1 = size(subsetA1,2);
subset_lengthA2 = size(subsetA2,2);
% DIFFERENT SUBSAMPLES 
% Choose amount of data to remove from each class (i.e amount of test data)
percentageA1 = 20; 
percentageA2 = 80;
percentageB = 0;
A1_train = subsetA1(:,1:(subset_lengthA1-(percentageA1/100)*subset_lengthA1));
A1_test = subsetA1(:,(subset_lengthA1-(percentageA1/100)*subset_lengthA1)+1:subset_lengthA1);
A2_train = subsetA2(:,1:(subset_lengthA2-(percentageA2/100)*subset_lengthA2));
A2_test = subsetA2(:,(subset_lengthA2-(percentageA2/100)*subset_lengthA2)+1:subset_lengthA2);

B_train = dataB_shuffled(:,1:(ndata-percentageB));
B_test = dataB_shuffled(:,(ndata-percentageB+1):ndata);


A_train = [A1_train,A2_train];
A_test = [A1_test,A2_test];
data_train = [A_train, B_train];
data_test = [A1_test,A2_test, B_test];
% data_train_shuffled = data_train(:,randperm(size(data_train,2)));
% data_test_shuffled = data_test(:,randperm(size(data_test,2)));
X_train = [data_train(1:2,:); ones(1,size(data_train,2))]; % +bias
T_train = data_train(3,:);
X_test = [data_test(1:2,:); ones(1,size(data_test,2))];    % +bias
T_test = data_test(3,:);


[row, col] = size(data_train);
w = randn(row,1);
w = transpose(w);
ndata_A = size(A_train,2);
ndata_B = size(B_train,2);
ndata_At = size(A_test,2);
ndata_Bt = size(B_test,2);
%block
figure(1);
[mse_block,weights_training] = delta_rule_block(X_train,T_train,w,data_train);

%Error
 [totalerrorA_training, totalerrorB_training] = calculate_error(weights_training, X_train, T_train);
 error_ratioA_training = totalerrorA_training/(ndata_A);
 error_ratioB_training = totalerrorB_training/(ndata_B);

title('DELTA RULE BLOCK IMPLEMENTATION - DECISION BOUNDARY');


%TESTING DATA

%Error
 [totalerrorA_test, totalerrorB_test] = calculate_error(weights_training, X_test, T_test);
 error_ratioA_test = totalerrorA_test/(ndata_At);
 error_ratioB_test = totalerrorB_test/(ndata_Bt);


%boundary(dataset,w);
n=1:1:20;
figure(3);
plot(n,mse_block);

grid on;
title('Mean squared Error vs Number of epochs');
xlabel('Number of epochs');
ylabel('Mean squared Error');



function [mse,w]=delta_rule_block(trainingdata,target,w,dataset)
   for k = 1 : 20 % No of Iterations
       squared_error=zeros(1,200);
     delta_w_vec = zeros(3,1); % Initialize

          Y = w*trainingdata;
    delta_W = -0.001*(Y-target)*trainingdata'; 
    squared_error = (Y-target).^2;
    w = w + delta_W;
     mse(k) = sum(squared_error)/200;
     boundary(trainingdata,w,dataset);
    
    end
    
end


function scatterplot(datasetplot)
%plot decision Boundary
set(gcf, 'Position', get(0,'Screensize')); 
gscatter(datasetplot(1,:),datasetplot(2,:),datasetplot(3,:),[],[],[],'off');
hold on;
end

    
 function boundary(datasetplot,w, dataset)
% Create the Decision Boundary
w1 = w(1);
w2 = w(2);
b = w(3);

clf

axis([-5 5 -10 10])

scatterplot(dataset);

xi = linspace(min(datasetplot(1,:)), max(datasetplot(1,:)));
yi=(-w1/w2)*xi + (-b/w2);
plot(xi, yi,'linewidth',1.5, 'color','black');
drawnow

legend('Class - 1',' Class - 2','Decision Boundary');
xlabel('F1');
ylabel('F2');
 end


 function [totalerrorA, totalerrorB] = calculate_error(W, X, T)
   Y = W*X;
   Y(Y>=0)=1; 
   Y(Y<0)=-1;
   check = [Y;T];
   len = (length(check));
   half = ceil(len/2); %for odd number of bit-stream length
   classA_separated = check(:,1:half);
   classB_separated = check(:,half + 1 : end);
   errorA = classA_separated(2,:)-classA_separated(1,:);  %T-Y
   errorB = classB_separated(2,:)-classB_separated(1,:);
   totalerrorB = nnz( errorA );
   totalerrorA = nnz( errorB );
%    sum_errA = sum(abs(errorA))/2;
%    sum_errB = sum(abs(errorB))/2;
 end