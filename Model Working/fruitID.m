%% Load image data and extract features

% Load dataset
imset = imageSet('trainingImages','recursive'); 

% Extract features
bag = bagOfFeatures(imset,'VocabularySize',200, 'PointSelection','Detector');

% Encode the images as new features
imagefeatures = encode(bag,imset);

% Create a Table using the encoded features
fruitData = array2table(imagefeatures);
fruitData.type = getLabels(imset);

%Use the new features to train a model and assess its performance
classificationLearner %% Do not run whole code, run section first, train the model
% and then after exporting it run the next part


%% Test Trained Model
img = imread('s.jpg');    % Just put image that we are trying to test here
    
% Step 2: Extract Features
imagefeatures = double(encode(bag,img));

% Step 3: Make prediction
tb = array2table(imagefeatures);
y = trainedModel.predictFcn(tb);
disp("This fruit is a:");
disp(y);
