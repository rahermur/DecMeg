% DecMeg2014 example code.
% Simple prediction of the class labels of the test set by:
%   - pooling all the triaining trials of all subjects in one dataset.
%   - Extracting the MEG data in the first 500ms from when the stimulus starts.
% - Using a linear classifier (elastic net).
% Implemented by Seyed Mostafa Kia (seyedmostafa.kia@unitn.it) and Emanuele
% Olivetti (olivetti@fbk.eu) as a benchmark for DecMeg 2014.
clear all;

rng(43); 

run('/Users/rafa/Dropbox/KAGGLE/DecMeg/matlab/adding_path.m')

Lou = randperm(16); 
acc = zeros(1,length(Lou)); 

sensores = 1:306; 
sensores = ~ismember(sensores,[173 73 25]); 
trainsamples = 1:16; 

for k = 1:length(Lou)

    disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');
    subjects_train = Lou(~ismember(trainsamples,k)) ;    
    disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));
    % We throw away all the MEG data outside the first 0.5sec from when
    % the visual stimulus start:
    tmin = 0;
    tmax = 0.5;
    disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
    X_train = [];
    y_train = [];
    X_test = [];
    ids_test = [];
    % Crating the trainset. (Please specify the absolute path for the train data)
    disp('Creating the trainset.');
    for i = 1 : length(subjects_train)
        path = './data/';  % Specify absolute path
        filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_train(i));
        disp(strcat('Loading ',filename));
        data = load(filename);
        XX = data.X;
        
        XX = XX(:,sensores,:); 
        yy = data.y;
        sfreq = data.sfreq;
        tmin_original = data.tmin;
        disp('Dataset summary:')
        disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
        disp(sprintf('yy: %d trials',size(yy,1)));
        disp(strcat('sfreq:', num2str(sfreq)));
        features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    end
    %
    % Crating the testset. (Please specify the absolute path for the test data)
    disp('Creating the testset.');
    X_val = [];
    y_val = [];
    subjects_val = Lou(k);
    for i = 1 : length(subjects_val)
        path = './data/'; % Specify absolute path
        filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_val(i));
        disp(strcat('Loading ',filename));
        data = load(filename);
        XX = data.X;
        yy = data.y;
        sfreq = data.sfreq;
        tmin_original = data.tmin;
        disp('Dataset summary:')
        disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
        disp(strcat('sfreq:', num2str(sfreq)));
        features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
%         
%         [COEFF, SCORE, LATENT] = pca(features);
%         features = features*COEFF(:,1:120);
        X_val = [X_val;features];
        y_val = [y_val;yy];
    end

    
    %%
    [COEFF, SCORE, LATENT] = pca(X_train);

    %%
    X_tr= X_train*COEFF(:,1:2000);

    X_v = X_val*COEFF(:,1:2000);


    %%
    % Your training code should be here:
    disp('Training the classifier ...')
    [BFinal,FitInfoFinal] = lasso(X_tr,single(y_train),'Lambda',0.0045,'Alpha',0.2);

    % Testing the trained classifier on the test data
    y_pred = [ones(size(X_v,1),1) X_v] * [FitInfoFinal.Intercept;BFinal];
    y_pred_thresholded = zeros(size(y_pred));
    y_pred_thresholded(y_pred>=median(y_pred))= 1;

    acc = sum(y_pred_thresholded==y_val)/length(y_pred);
    disp(['ACC ', num2str(acc)])

    % Your training code should be here:
    disp('Training the classifier ...')
    [BFinal,FitInfoFinal] = lasso(X_train,single(y_train),'Lambda',0.0045,'Alpha',0.2);

    % Testing the trained classifier on the test data
    y_pred = [ones(size(X_val,1),1) X_val] * [FitInfoFinal.Intercept;BFinal];
    y_pred_thresholded = zeros(size(y_pred));
    y_pred_thresholded(y_pred>=median(y_pred))= 1;

    acc(k) = sum(y_pred_thresholded==y_val)/length(y_pred);
    disp(['ACC ', num2str(acc(k))])

end

disp(['Final Leave One Out Accuracy', num2str(mean(acc))])