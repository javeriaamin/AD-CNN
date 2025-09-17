function augment_datasets_balanced(srcBase, dstBase)
    % Dataset info: {name, numClasses, beforeAug, afterAug}
    datasets = {
        'ADNI-1', 4, 6400, 20880;
        'ADNI-2', 5, 1296, 37175;
        'OASIS-2', 2, 17365, 45000
    };

    for d = 1:size(datasets,1)
        datasetName   = datasets{d,1};
        numClasses    = datasets{d,2};
        origCount     = datasets{d,3};
        targetCount   = datasets{d,4};

        fprintf('\n=== Processing %s ===\n', datasetName);
        fprintf('Classes: %d | Before: %d | After: %d\n', ...
            numClasses, origCount, targetCount);

        % Dataset folders
        srcDir = fullfile(srcBase, datasetName);
        dstDir = fullfile(dstBase, datasetName);
        if ~exist(dstDir, 'dir'), mkdir(dstDir); end

        % Collect per-class images
        classFolders = dir(srcDir);
        classFolders = classFolders([classFolders.isdir] & ~ismember({classFolders.name},{'.','..'}));

        % Check class count
        if length(classFolders) ~= numClasses
            warning('%s: Expected %d classes, found %d folders', datasetName, numClasses, length(classFolders));
        end

        % Compute per-class target counts (balanced)
        perClassTarget = floor(targetCount / numClasses);
        remainder = targetCount - perClassTarget*numClasses;

        fprintf('Per-class target ~ %d (+ distribute %d remainder)\n', perClassTarget, remainder);

        for c = 1:length(classFolders)
            className = classFolders(c).name;
            classSrc = fullfile(srcDir, className);
            classDst = fullfile(dstDir, className);
            if ~exist(classDst, 'dir'), mkdir(classDst); end

            % Load images in this class
            imgExt = {'*.jpg','*.png','*.jpeg'};
            classFiles = [];
            for i = 1:length(imgExt)
                classFiles = [classFiles; dir(fullfile(classSrc, imgExt{i}))];
            end
            Nc = length(classFiles);

            % Class target (add remainder to first few classes)
            Tc = perClassTarget + (c <= remainder);
            fprintf('  %s: %d originals → %d target\n', className, Nc, Tc);

            % Augmentation plan
            avgFactor = Tc / Nc;
            floorF = floor(avgFactor);
            ceilF  = ceil(avgFactor);

            numHigh = Tc - (floorF * Nc);
            numLow  = Nc - numHigh;

            fprintf('    Aug plan: %d imgs→%d augs, %d imgs→%d augs\n', ...
                numHigh, ceilF, numLow, floorF);

            % Process each image
            for idx = 1:Nc
                imgPath = fullfile(classFiles(idx).folder, classFiles(idx).name);
                img = imread(imgPath);
                if size(img,3) == 1
                    img = repmat(img, [1 1 3]);
                end

                [~, baseName, ext] = fileparts(imgPath);

                % Save original
                imwrite(img, fullfile(classDst, [baseName '_orig' ext]));

                % Decide augmentation count
                if idx <= numHigh
                    augCount = ceilF;
                else
                    augCount = floorF;
                end

                % Apply augmentations
                for a = 1:augCount
                    augImg = apply_random_aug(img);
                    outPath = fullfile(classDst, sprintf('%s_aug%d%s', baseName, a, ext));
                    imwrite(augImg, outPath);
                end
            end
        end

        fprintf('%s done. Balanced augmentation complete.\n', datasetName);
    end
end

function augImg = apply_random_aug(img)
    augImg = img;

    % Flip
    if rand > 0.5, augImg = fliplr(augImg); end
    if rand > 0.7, augImg = flipud(augImg); end

    % Rotation
    if rand > 0.5
        angle = randi([-15, 15]);
        augImg = imrotate(augImg, angle, 'crop');
    end

    % Brightness & contrast
    if rand > 0.5
        augImg = imadjust(augImg, [], [], 0.8 + 0.4*rand);
    end

    % Gaussian noise
    if rand > 0.3
        augImg = imnoise(augImg, 'gaussian', 0, 0.005*rand);
    end

    % Resize
    augImg = imresize(augImg, [size(img,1), size(img,2)]);
end
