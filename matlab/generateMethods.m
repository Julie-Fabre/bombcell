%% bombcell: Methods Text Generation
% This script demonstrates how to automatically generate a methods section
% paragraph for a scientific paper, based on the parameters used to run
% bombcell

%% 1. Generate methods text from default parameters
% If you just want to quickly generate the text with default BombCell
% parameters, you can create a minimal param struct. No data is needed.

param = bc.qm.qualityParamValues([], 'NaN', '', 1);

% Generate and print everything
bc.qm.printMethodsText(param);

%% 2. Generate from your actual analysis parameters
% If you've already run bombcell on your data, load your saved parameters
% and generate from those:

%%% Option A: Load saved parameters from a previous bombcell run
% load(fullfile(savePath, 'qMetric.mat'), 'param');
% bc.qm.printMethodsText(param);

%%% Option B: Use parameters you've configured in this session
% ephysMetaDir = dir(fullfile(myDataPath, '*.ap.meta'));
% rawFile = fullfile(myDataPath, 'myRecording.ap.bin');
% ksPath = '/path/to/kilosort/output';
% param = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ksPath);

% Customize parameters as you would for your analysis
param.tauR_valuesMin = 0.5 / 1000;
param.tauR_valuesMax = 5 / 1000;
param.tauR_valuesStep = 0.5 / 1000;
param.computeDrift = 1;
param.computeDistanceMetrics = 0;
param.computeTimeChunks = 1;
param.maxRPVviolations = 0.1;
param.minAmplitude = 50;

% Generate with inline citations (default)
bc.qm.printMethodsText(param);

%% 3. Citation styles
% You can choose between inline author-year citations or numbered
% references:

% Numbered citation style
bc.qm.printMethodsText(param, 'citationStyle', 'numbered');

