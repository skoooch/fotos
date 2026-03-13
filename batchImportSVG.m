function allVecLDs = batchImportSVG(svgFolder, outputMatFile, imsize)
% allVecLDs = batchImportSVG(svgFolder, outputMatFile, imsize)
% Loops through all SVG files in a folder, imports each into a vecLD
% structure, and saves all of them into a single .mat file.
%
% Input:
%   svgFolder    - path to folder containing SVG files
%   outputMatFile - path for the output .mat file (e.g. 'vecLDs.mat')
%   imsize       - [optional] image size [width, height] to use for all SVGs
%
% Output:
%   allVecLDs - struct array with fields:
%       .filename - the SVG filename
%       .vecLD    - the vecLD structure from importSVG

if nargin < 2 || isempty(outputMatFile)
    outputMatFile = fullfile(svgFolder, 'vecLDs.mat');
end

if nargin < 3
    imsize = [];
end

% Get all SVG files in the folder
svgFiles = dir(fullfile(svgFolder, '*.svg'));

if isempty(svgFiles)
    error('No SVG files found in folder: %s', svgFolder);
end

fprintf('Found %d SVG files in %s\n', numel(svgFiles), svgFolder);

allVecLDs = struct('filename', {}, 'vecLD', {});

for i = 1:5
    svgPath = fullfile(svgFolder, svgFiles(i).name)
    fprintf('[%d/%d] Importing %s ...', i, numel(svgFiles), svgFiles(i).name);
    
    try
        if isempty(imsize)
            vecLD = importSVG(svgPath);
        else
            vecLD = importSVG(svgPath, imsize);
        end
        
        allVecLDs(end+1).filename = svgFiles(i).name; %#ok<AGROW>
        allVecLDs(end).vecLD = vecLD;
        fprintf(' done (%d contours)\n', vecLD.numContours);
    catch ME
        fprintf(' ERROR: %s\n', ME.message);
    end
end

% Save to .mat file
save(outputMatFile, 'allVecLDs', '-v7.3');
fprintf('Saved %d vecLD structures to %s\n', numel(allVecLDs), outputMatFile);

end