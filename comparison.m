clc;
clear;
close all;

n = 100;
mA = [ 2.0, 1]; sigmaA = 0.5;
mB = [-2.0, -1]; sigmaB = 0.5;
classA(1,:) = randn(1,n) .* sigmaA + mA(1);
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

classA = [classA; ones(size(classA(1,:)))];
classB = [classB; -ones(size(classB(1,:)))];

dataset = horzcat(classA,classB);
%scatterplot(dataset);


%plotting(datasetplot,w,target);


[row, col] = size(dataset);
w = randn(row,1);
w = transpose(w);

%block
figure(1);
[mse_block,w] = delta_rule_block(dataset,w,col);
title('DELTA RULE BLOCK IMPLEMENTATION - DECISION BOUNDARY');

% %sequential
% figure(2);
% [mse_sequential,w] = delta_rule_sequential(dataset,w,col);
% title('DELTA RULE SEQUENTIAL IMPLEMENTATION - DECISION BOUNDARY');

%boundary(dataset,w);
n=1:1:20;
figure(3);
plot(n,mse_block);
% hold on;
% plot(n,mse_sequential);
% title('MSE delta rule block vs sequential mode');


function [mse,w]=delta_rule_block(dataset,w,col)

    for k = 1 : 20 % No of Iterations
    squared_error=zeros(1,200);
    index_random = randperm(length(dataset),length(dataset));
    datasetshuffle = dataset(:,index_random);
    target = datasetshuffle(3,:);
    datasetshuffle(3,:) = [];
    data_train = [datasetshuffle; ones(size(datasetshuffle(1,:)))];
    delta_w_vec = zeros(3,1); % Initialize

%         %matrix multiplcation method
%         error = target - w* data_train;
%         squared_error = error.^2;
%         delta_w_arr = 0.001.*((error)*data_train');
%         %delta_w_vec = sum(delta_w_arr,2);
%     % (CHECKED)
% %     if (target(i) == y(i))% Until all the samples are correctly classified
%      w = w + delta_w_arr; 


Y = w*data_train;
delta_W = -0.001*(Y-target)*data_train'; 
squared_error = (Y-target).^2;
w = w + delta_W;
     mse(k) = sum(squared_error)/200;
     boundary(data_train,w,dataset);
    
    end
    
end

function [mse,w]=delta_rule_sequential(dataset,w,col)

    for k = 1 : 20 % No of Iterations
    squared_error=zeros(1,200);
    index_random = randperm(length(dataset),length(dataset));
    datasetshuffle = dataset(:,index_random);
    target = datasetshuffle(3,:);
    datasetshuffle(3,:) = [];
    data_train = [datasetshuffle; ones(size(datasetshuffle(1,:)))];
    delta_w_vec = zeros(3,1); % Initialize

        %sequential
        for i = 1 : col % For all Samples
            % Compute the Weighted Sum
            z = w* data_train(:,i); % 
            % Checking Result (if update for weights required or not)
            % delta_w_vec = zeros(3,1); % Initialize
            error = target(i) - z;
            squared_error(i) = error.^2;
            if error ~= 0 % Not Equal Case
                % Perform Updation of weights
                delta_w = (0.001*error).*data_train(:,i);   
            else 
                % do nothing
            end
            w = w + delta_w';          
        end

     mse(k) = sum(squared_error)/200;
     boundary(data_train,w,dataset);
    
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
