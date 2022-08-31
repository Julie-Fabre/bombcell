%% run brainreg 

CMD=sprintf('brainreg %s.mhd %s.mhd -v 25 25 25 --orientation psl %s ',...
            fullfile(outputDir,targetFname),...
            fullfile(outputDir,movingFname),...
            outputDir);
CMD = [CMD,initCMD];


if ~isempty(threads)
    CMD = sprintf('%s -threads %d',CMD,threads);
end


%Loop through, adding each parameter file in turn to the string
for ii=1:length(paramFname) 
    CMD=[CMD,sprintf('-p %s ', paramFname{ii})];
end

%store a copy of the command to the directory
cmdFid = fopen(fullfile(outputDir,'CMD'),'w');
fprintf(cmdFid,'%s\n',CMD);
fclose(cmdFid);





% Run the command and report back if it failed
fprintf('Running: %s\n',CMD)

[status,result]=system(CMD);