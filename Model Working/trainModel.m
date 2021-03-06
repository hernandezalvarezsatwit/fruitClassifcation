%This function uses the training data set to extract features and create
%a model
function [trainedModel, bag] = trainModel()

    %TODO: REPLACE TRAINING IMAGES WITH FOLDER WITH PREPROCESSED AND SEGMENTED
    %IMAGES
    imset = imageSet('trainingImages','recursive');  % Load Image Folders
    
    % Extract features
    bag = bagOfFeatures(imset,'VocabularySize',500, 'PointSelection','Detector');

    % Encode the images as new features
    imagefeatures = encode(bag,imset);

    % Create a Table using the encoded features
    fruitData = array2table(imagefeatures);
    fruitData.type = getLabels(imset);
    
    % Train the model using the data set. 
    % TODO: CONSDIER ADJUSTING THE MACHINE LEARNING TECHNIQUE USED
    % ONCE WE CAN COMPARE PEFORMANCES WITH PROCESSED TRAINING IMAGES
    trainedModel = trainClassifier(fruitData);
    
    clc; 
end