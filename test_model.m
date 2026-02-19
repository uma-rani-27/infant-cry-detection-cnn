clc; 
clear; 
close all;

% Load trained model and cry categories
load('babycry_trained_model.mat');
fprintf("✅ Model loaded successfully!\n");

% Open file selection dialog
[filename, pathname] = uigetfile({'*.wav'}, ...
    'Select a baby cry audio file');

% Check if user selected a file
if isequal(filename, 0)
    disp('❌ No file selected.');
    return;
end

file = fullfile(pathname, filename);
fprintf("🎧 Testing file: %s\n", file);

% Read audio file
[audio, fs] = audioread(file);

% Convert stereo to mono if required
if size(audio, 2) > 1
    audio = mean(audio, 2);
end

% Resample audio to 16 kHz if needed
if fs ~= 16000
    audio = resample(audio, 16000, fs);
    fs = 16000;
end

% Extract MFCC features using custom function
features = extract_features(audio, fs);

% Predict cry category using trained SVM model
label_pred = predict(model, features);

fprintf("\n🔍 Predicted Emotion: %s\n", string(label_pred));

% Play audio sound
sound(audio, fs);
