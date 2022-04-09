%% Load image data and extract features

% Load dataset
imset = imageSet('trainingImages','recursive');  % Change to trainingImages

% Extract features
bag = bagOfFeatures(imset,'VocabularySize',500, 'PointSelection','Detector');
%bag = bagOfFeatures(imset);

% Encode the images as new features
imagefeatures = encode(bag,imset);

% Create a Table using the encoded features
fruitData = array2table(imagefeatures);
fruitData.type = getLabels(imset);

%Use the new features to train a model and assess its performance
classificationLearner %% Do not run whole code, run section first, train the model
% and then after exporting it run the next part


%% Test Trained Model
%img = imread('/Users/samuelmac/Desktop/AppleTest/_5.jpg');    % Just put image that we are trying to test here
    
% Step 2: Extract Features
%imagefeatures = double(encode(bag,img));

% Step 3: Make prediction
%tb = array2table(imagefeatures);
%y = trainedModel.predictFcn(tb);
%disp("This fruit is a:");
%disp(y);
%test(trainedModel, bag, "b.jpg")
%Display success rate on test observations
%[results, successRate] = testModel(trainedModel, bag, 1, 248); 
%disp(successRate);

%disp(results);


%% Function runs over the test data and returns the success rate as
% well as an array indicatin the classified prediction of the algorithm
% Parameters:
% trainedModel = the model to use to test
% bag = the bag used to test the model
% y = the expected result 1 for apple 2 for orange
% obs = the number of available observations to test on
% 
% Format the path to the test data directory
function [results, error] = testModel(trainedModel, bag, y, obs)
    results = 1*obs;
    max = obs;
    ok = 0;
    for i = 1 : max
        %Format path to location of test data
        path = 'testImages/AppleTest/_';
        path = strcat(path, string(i));
        path = strcat(path,'.jpg');
        img = imread(path);
        imagefeatures = encode(bag, img);
        tb = array2table(imagefeatures);
        yhat = trainedModel.predictFcn(tb);
        results(i) = yhat;
        if results(i) == y
            ok = ok+1;
        end
    end 
    error = ok/max;
end

%%
function y = test(trainedModel, bag, path)
    img = imread(path);   
    imagefeatures = double(encode(bag,img));
    tb = array2table(imagefeatures);
    y = trainedModel.predictFcn(tb);
    disp("This fruit is a:");
    disp(y);
end
