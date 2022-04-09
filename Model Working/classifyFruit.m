function classifyFruit(path, trainedModel, bag)
    img = imread(path); 
    imagefeatures = double(encode(bag,img));
    tb = array2table(imagefeatures);
    y = trainedModel.predictFcn(tb);
    disp("This fruit is a:");
    disp(y);
end
