%% Load image data and extract features

% Step 1. Load dataset
% Parameter trainData: name of folder with all training data divided 
% into folders labeled with each fruit's name. Recursive adds all
% directories withign folder. 
dataSet = imageSet('trainData','recursive'); 

% Step 2. Extract features
% Function extracts features from the data set
bag = bagOfFeatures(dataSet,'VocabularySize', 500, 'PointSelection','Detector');

% Step 3. Encode the images as new features
imagefeatures = encode(bag,dataSet);

% Create a Table using the encoded features
fruitData = array2table(imagefeatures);
fruitData.type = getLabels(dataSet);

%Use the new features to train a model and assess its performance
%Once the classification learner opens click ok:
% New Session> Select Input > fruitData > Start Session > Click on the
% Machine Learning Modesl > Select Medium Gaussian > Train > Once it is
% done training > Export > name = trainedModel > Run next section
classificationLearner


%% Test trained model on test data

%Display success rate on test observations
[results, successRate] = testModel(trainedModel, bag, 2, 248); 
disp(successRate);


%% Function runs over the test data and returns the success rate as
% well as an array indicating the classified prediction of the algorithm
% Parameters:
% trainedModel = the model to use to test
% bag = the bag used to test the model
% y = the expected result 1 for apple 2 for orange
% obs = the number of available observations to test on
% 
% Format the path to the test data directory
% Return values-> results is an array containing the classifications. 
% 1 if the model classified it as apple or 2 if as orange
function [results, successRate] = testModel(trainedModel, bag, y, obs)
    results = 1*obs;    
    max = obs;
    correct = 0;
    for i = 1 : max
        %Format path to location of test data
        path = 'testImages/OrangeTest/_';
        path = strcat(path, string(i),'.jpg');
        imagefeatures = encode(bag, imread(path));
        tb = array2table(imagefeatures);
        yhat = trainedModel.predictFcn(tb);
        results(i) = yhat;
        if results(i) == y
            correct = correct+1;
        end
    end 
    successRate = correct/max;
end


% Next steps
% Try using the huge train set and see if error rate can be better than 
% 74~79% on classificationLearner or 75%  in test data
% Also, use small training set to train model and see if training with
% boundaries only help