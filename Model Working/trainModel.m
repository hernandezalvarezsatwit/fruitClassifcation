function [trainedModel, bag] = trainModel()
    imset = imageSet('trainingImages','recursive');  % Change to trainingImages
    
    % Extract features
    bag = bagOfFeatures(imset,'VocabularySize',500, 'PointSelection','Detector');

    % Encode the images as new features
    imagefeatures = encode(bag,imset);

    % Create a Table using the encoded features
    fruitData = array2table(imagefeatures);
    fruitData.type = getLabels(imset);
    trainedModel = trainClassifier(fruitData);
end