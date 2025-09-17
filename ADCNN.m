digitDatasetPath = 'E:\augmented\ADNI-1';

% Image datastore
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Preprocess: resize to 256x256, keep RGB
imds.ReadFcn = @(filename) imresize(imread(filename), [256 256]);

% CNN Layers
imageSize = [256 256 3];
layers = [
    imageInputLayer(imageSize)

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer('Name','relu1')   

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer('Name','relu2')   

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer('Name','relu3')   

    fullyConnectedLayer(numel(unique(imds.Labels))) % auto detect classes
    softmaxLayer
    classificationLayer];

% Training options
opts = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train CNN
net = trainNetwork(imds, layers, opts);

%% 🔹 Extract features
featureLayer = 'relu3';  
allFeatures = activations(net, imds, featureLayer, 'OutputAs','rows');
allLabels   = grp2idx(imds.Labels);  

%% 🔹 Arithmetic Optimization Algorithm (ASO) for Feature Selection
% Parameters
lb = 0;                % lower bound
ub = 1;                % upper bound
thresh = 0.5;          % threshold to select feature
alpha = 50;            % α
beta = 0.2;            % β
Nsol = 10;             % total solutions
MaxIter = 100;         % max iterations
dim = size(allFeatures,2);  % number of features

% Initialize population
pop = lb + (ub-lb).*rand(Nsol, dim);

% Best solution placeholder
bestSol = pop(1,:);
bestFit = 0;

% Main ASO loop (without classifier, just optimizing randomly)
for it=1:MaxIter
    for i=1:Nsol
        % Arithmetic update rule
        A = alpha * (1 - it/MaxIter); 
        B = beta * randn(1,dim); 
        newSol = pop(i,:) + A*rand(1,dim) - B.*pop(i,:);
        
        % Bound control
        newSol = min(max(newSol,lb),ub);
        
        % Random fitness proxy = number of selected features (maximize diversity)
        sel = newSol > thresh;
        fitVal = sum(sel);   % here fitness = total features selected
        
        % Greedy selection
        if fitVal > bestFit
            bestFit = fitVal;
            bestSol = newSol;
        end
    end
    
    fprintf('Iteration %d/%d - Selected Features: %d\n', it, MaxIter, bestFit);
end

%% 🔹 Final Selected Features
selectedFeatures = allFeatures(:, bestSol > thresh);
labels = allLabels;

% Save only optimized features and labels
save('Optimized_Features_ASO.mat','selectedFeatures','labels');

disp(' Only optimized features + labels saved to Optimized_Features_ASO.mat');
