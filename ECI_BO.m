clearvars;clc;close all;
% objective function name
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of initial design points
num_initial = 2*num_vari;
% number of maximum evaluations
max_evaluation = 10*num_vari;
% initial design
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
iteration = 1;
evaluation =  size(sample_x,1);
[fmin,ind] = min(sample_y);
best_x = sample_x(ind,:);
fmin_record(iteration,1) = fmin;
fprintf('ECI-BO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
remain_dim = 1:num_vari;
while evaluation < max_evaluation
    % train GP models
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    % find the maximum ECI values for all the coordinates
    max_ECI = zeros(1,num_vari);
    max_x = zeros(1,num_vari);
    % if you have multiple cores, you can use parfor to compute this in
    % parallel
    for ii = 1:num_vari
        [max_x(ii),max_ECI(ii)] = Optimizer_GA(@(x)-Infill_ECI(x,GP_model,fmin,best_x,ii),1,lower_bound(ii),upper_bound(ii),10,20);
    end
    % get the coordinate optimization order based on the ECI values
    [sort_EI,sort_dim] = sort(-max_ECI,'descend');
    for ii = 1:num_vari
        % optimize one coordinate at a time following the above coordinate
        % optimization order
        select_dim = sort_dim(ii);
        if ii == 1
            candidate_x = max_x(select_dim);
        else
            GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
            [candidate_x,EI] = Optimizer_GA(@(x)-Infill_ECI(x,GP_model,fmin,best_x,select_dim),1,lower_bound(select_dim),upper_bound(select_dim),10,20);
        end
        % get a new solution
        infill_x = best_x;
        infill_x(:,select_dim) = candidate_x;
        % evaluate the new solution
        infill_y = feval(fun_name,infill_x);
        iteration = iteration + 1;
        sample_x = [sample_x;infill_x];
        sample_y = [sample_y;infill_y];
        [fmin,ind] = min(sample_y);
        best_x = sample_x(ind,:);
        fmin_record(iteration,1) = fmin;
        evaluation = evaluation + size(infill_x,1);
        fprintf('ECI-BO on %d-D %s, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,iteration-1,evaluation,fmin);
    end
end

