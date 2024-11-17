
%% raw data GUI

function rawDataGuiHandle = rawDataGUI(memMapData, ephysData, plotColor)

% if isempty(memMapData)
%     display('Extract data on the fly using python. You need Matlab version 2022a or higher, and have it pointed to the correct (Anaconda) environment')
% end

%% set up dynamic figure
rawDataGuiHandle = figure('color', plotColor);
set(rawDataGuiHandle, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
uniqueTemps = unique(ephysData.spike_templates);
[~, maxSite] = max(max(abs(ephysData.templates), [], 2), [], 3);
   
%% plot initial conditions
iChunk = 1;
iChannelChunk = 1;
initializePlot(rawDataGuiHandle, plotColor)
updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
    uniqueTemps, maxSite, iChunk, iChannelChunk)
%% change on keypress
    function KeyPressCb(~, evnt)
        %fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'leftarrow')
            iChunk = iChunk + 1;
            if iChunk > length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster))
                iChunk = 1;
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
                uniqueTemps, maxSite, iChunk, iChannelChunk);
        elseif strcmpi(evnt.Key, 'rightarrow')
            iChunk = iChunk - 1;
            if iChunk == 0
                iChunk = length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster));
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
                uniqueTemps, maxSite, iChunk, iChannelChunk);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChannelChunk = iChannelChunk + 1;
            if iChannelChunk > floor(384/20)
                iChannelChunk = iChannelChunk - 1;
            end
            updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
                uniqueTemps, maxSite, iChunk, iChannelChunk);

        elseif strcmpi(evnt.Key, 'downarrow')
            iChannelChunk = iChannelChunk - 1;
            if iChannelChunk < 1
                iChannelChunk = iChannelChunk + 1;
            end
           updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
                uniqueTemps, maxSite, iChunk, iChannelChunk);
        end
    end
end

function initializePlot(rawDataGuiHandle, plotColor)

%% main title

mainTitle = sgtitle('');

%% initialize raw data
if plotColor == 'k'
    lineColor = 'w';
else
    lineColor = 'k';
end
max_n_channels_plot = 20;
rawPlotH = subplot(1, 1, 1);
set(gca,'color',[0 0 0])
hold on;
title('Raw unwhitened data')
set(rawPlotH, 'XColor', plotColor, 'YColor', plotColor)
rawPlotLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', lineColor), 1:max_n_channels_plot);
allUnitColors = viz.colors(100, plotColor);
for iUnit = 1:100 %max 30 units in 20 channels?
    if exist("rawSpikeLines", 'var')
        rawSpikeLines = [rawSpikeLines, arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', allUnitColors(iUnit,:)), 1:max_n_channels_plot)];
    else
        rawSpikeLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color',  allUnitColors(iUnit,:)), 1:max_n_channels_plot);
    end
end
%% save all handles
guiData = struct;
% main title
guiData.mainTitle = mainTitle;

% raw data

guiData.rawPlotH = rawPlotH;
guiData.rawPlotLines = rawPlotLines;
guiData.rawSpikeLines = rawSpikeLines;

% upload guiData
guidata(rawDataGuiHandle, guiData);
end

function updateRawSnippet(rawDataGuiHandle, memMapData, ephysData, ...
    uniqueTemps, maxSite, iChunk, iChannelChunk)


guiData = guidata(rawDataGuiHandle);

%% 8. plot raw data

plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, memMapData, ephysData, maxSite, uniqueTemps, iChunk, iChannelChunk);


end


function plotSubRaw(rawPlotH, rawPlotLines, rawSpikeLines, memMapData, ephysData, maxSite, uniqueTemps, iChunk, iChannelChunk)
%get the used channels
chansToPlot = (iChannelChunk - 1)* 20 + 1: (iChannelChunk )* 20;
realMaxSite = ismember(1:length(maxSite), uniqueTemps);
theseUnits = find(ismember(maxSite(realMaxSite)-4, chansToPlot) | ismember(maxSite(realMaxSite)+4, chansToPlot));
% get units with max channel displayed
theseTimesFull = [];
thisUnitFull = [];
theseTimesFullCenter = [];
for iCluster = 1:length(theseUnits)

    %get spike locations
    timeToPlot = 0.1;
    pull_spikeT = -40:41;
    thisC = uniqueTemps(theseUnits(iCluster));

    theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC) ./ ephysData.ephys_sample_rate;
    if iChunk < 0
        disp('Don''t do that')
        iChunk = 1;
    end
    if length(theseTimesCenter) > 10 + iChunk && iCluster == 1
        firstSpike = theseTimesCenter(iChunk+10) - 0.05; %tenth spike occurance %
    elseif iCluster == 1
        firstSpike = theseTimesCenter(iChunk) - 0.05; %first spike occurance
    end
    % Not sure why this was +10?
    theseTimesCenter = theseTimesCenter(theseTimesCenter >= firstSpike);
    theseTimesCenter = theseTimesCenter(theseTimesCenter <= firstSpike+timeToPlot);
    if ~isempty(theseTimesCenter)
        %theseTimesCenter=theseTimesCenter(1);
        theseTimesFull = [theseTimesFull; theseTimesCenter * ephysData.ephys_sample_rate + pull_spikeT];
        theseTimesFullCenter = [theseTimesFullCenter; theseTimesCenter];
        thisUnitFull = [thisUnitFull; ones(length(theseTimesCenter),1)*iCluster];
        %theseTimesFull=unique(sort(theseTimesFull));
    end

end
cCount = cumsum(repmat(1000, 20, 1), 1);


t = int32(firstSpike*ephysData.ephys_sample_rate):int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
subplot(rawPlotH)
plotidx = int32(firstSpike*ephysData.ephys_sample_rate) ...
    :int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
t(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
plotidx(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
thisMemMap = double(memMapData(chansToPlot, plotidx)) + double(cCount);
for iClear = 1:length(rawSpikeLines)
    set(rawSpikeLines(iClear), 'XData', NaN, 'YData', NaN)
end
if length(rawSpikeLines) < length(chansToPlot)
    rawSpikeLines(end+1:length(chansToPlot)) = rawSpikeLines(end);
    rawPlotLines(end+1:length(chansToPlot)) = rawPlotLines(end);
end

for iChanToPlot = 1:length(chansToPlot)
    hp_memmap = highpass(thisMemMap(iChanToPlot, :), 200, 30000)+200*(iChanToPlot-1);
    set(rawPlotLines(iChanToPlot), 'XData', t, 'YData', hp_memmap);

end
for iChanToPlot = 1:length(chansToPlot)
    hp_memmap = highpass(thisMemMap(iChanToPlot, :), 200, 30000)+200*(iChanToPlot-1);
    set(rawPlotLines(iChanToPlot), 'XData', t, 'YData', hp_memmap);
    for iUnit = 1:length(theseUnits)
        if ~isempty(theseTimesFullCenter(thisUnitFull==iUnit))
            thisUnitTimes = theseTimesFullCenter(thisUnitFull==iUnit);
            thisUnitTimesFull = theseTimesFull(thisUnitFull==iUnit,:);
            for iTimes = 1:size(thisUnitTimes, 1)
                if any(int32(thisUnitTimesFull(iTimes, :))-t(1) > length(t))
                else
                if  max(abs(hp_memmap(:, ...
                        int32(thisUnitTimesFull(iTimes, :))-t(1)))) - 200*(iChanToPlot-1) > 18
                    set(rawSpikeLines(iChanToPlot+(iUnit-1)*20), 'XData', thisUnitTimesFull(iTimes, :), 'YData', hp_memmap(:, ...
                        int32(thisUnitTimesFull(iTimes, :))-t(1)));
                end
                end
            end
        end
    end

end


end
