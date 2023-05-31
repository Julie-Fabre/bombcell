function paramEP = bc_ephysPropValues
% 
% JF, Load a parameter structure defining extraction and
% classification parameters
% ------
% Inputs
% ------
%
% ------
% Outputs
% ------
% paramEP: matlab structure defining ephys properties extraction and
%   classification parameters
% 

paramEP = struct; 
paramEP.plotThis = 0;
paramEP.verbose = 1;

paramEP.ephys_sample_rate = 30000;

paramEP.ACGbinSize = 0.001;
paramEP.ACGduration = 1;

paramEP.longISI = 2;


% QQ set this the same as qmetrics (load useChunks Start and stop)
%paramEP.computeTimeChunks = 1; % compute ephysProperties for different time chunks 
%paramEP.deltaTimeChunk = 360; %time in seconds 

% cell classification parameters
paramEP.propISI = 0.1;
paramEP.templateDuration = 400;
paramEP.pss = 40;

end
