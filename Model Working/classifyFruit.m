%Function reads in an image and uses the trained model to classify it
%The trained model and the bag of features of the model must be passed as
%parameters. 
function y = classifyFruit(path, trainedModel, bag)
    img = imread(path);                      %Load image from full path
    
    %TODO: COMPLETE THIS FUNCTION FOR PREPROCESSING AND SEGEMENTATION
    %FOLLOWING SAME PROCESS USED FOR TRAINING DATA
    img = processImage(img);
    
    imagefeatures = double(encode(bag,img)); %Extract Features
    tb = array2table(imagefeatures);         %As table
    y = trainedModel.predictFcn(tb);         %Make prediction
    clc;                                     %Clear console
end

