close all; clc; clear;

suitimages = imageDatastore(cd, 'IncludeSubfolders', true, 'LabelSource', 'FolderNames');

numTrainFiles = 900;
[TrainImages, TestImages] = splitEachLabel(suitimages, numTrainFiles, 'randomize');

layers = [
    imageInputLayer([22 15 1], 'Name', 'Input');
    
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'Conv_1');
    batchNormalizationLayer('Name', 'BN_1');
    reluLayer('Name', 'Relu_1');
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_1');
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv_2');
    batchNormalizationLayer('Name', 'BN_2');
    reluLayer('Name', 'Relu_2');
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_2');
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'Conv_3');
    batchNormalizationLayer('Name', 'BN_3');
    reluLayer('Name', 'Relu_3');
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'MaxPool_3');
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'Conv_4');
    batchNormalizationLayer('Name', 'BN_4');
    reluLayer('Name', 'Relu_4');
   
    fullyConnectedLayer(4, 'Name', 'FC');
    softmaxLayer('Name', 'SoftMax');
    classificationLayer('Name', 'Output Classification');
    ]

%lgraph = layerGraph(layers);
%plot(lgraph);

options = trainingOptions('sgdm', 'InitialLearnRate', 0.01, 'MaxEpoch', 4, 'Shuffle', 'every-epoch', 'ValidationData',...
    TestImages, 'ValidationFrequency', 30, 'Verbose', false, 'Plots', 'training-progress');

suit_net = trainNetwork(TrainImages, layers, options);
save suit_net;

Ypred = classify(suit_net, TestImages);
Yvalidation = TestImages.Labels;
accuracy = sum(Ypred == Yvalidation) / numel(Yvalidation);

