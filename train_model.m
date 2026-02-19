clc; 
clear; 
close all;

disp("Loading extracted features...");

% Load pre-extracted features and labels
load('babycry_features.mat');

disp("Training the classifier...");

% Convert labels into categorical format
all_labels = categorical(all_labels);

% -------- Dataset Balancing --------
% Ensure equal samples for each class
minCount = min(countcats(all_labels));
balanced_features = [];
balanced_labels = [];

cats = categories(all_labels);

for i = 1:length(cats)
    % Find indices of each category
    idx = find(all_labels == cats{i});

    % Randomly select equal samples
    idx = idx(randperm(length(idx), minCount));

    % Append balanced data
    balanced_features = [balanced_features; all_features(idx, :)];
    balanced_labels = [balanced_labels; all_labels(idx)];
end

disp("Dataset balanced successfully!");
tabulate(balanced_labels)

% -------- Train-Test Split --------
cv = cvpartition(balanced_labels, 'HoldOut', 0.2);

XTrain = balanced_features(training(cv), :);
YTrain = balanced_labels(training(cv));

XTest = balanced_features(test(cv), :);
YTest = balanced_labels(test(cv));

% -------- Train SVM Classifier --------
model = fitcecoc(XTrain, YTrain);

YPred = predict(model, XTest);
acc_svm = mean(YPred == YTest) * 100;

fprintf("SVM Accuracy = %.2f%%\n", acc_svm);

% -------- Train KNN Classifier --------
MdlKNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 5);

YP_knn = predict(MdlKNN, XTest);
acc_knn = mean(YP_knn == YTest) * 100;

fprintf("KNN Accuracy = %.2f%%\n", acc_knn);

% Save trained models
save('babycry_trained_model.mat', 'model', 'MdlKNN', 'cats');

disp("✅ Models saved successfully!");