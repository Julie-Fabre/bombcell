 function dynamicRawPlot(memMapData, ephysData)
    h = figure; 
    set(h,'KeyPressFcn',@KeyPressCb) ;
    iCluster=1;
    iCount=1;
    timeSecs = 1;
    timeChunkStart = 5000;
    timeChunk = timeSecs*ephysData(iCount).ephys_sample_rate;

    timeChunkStop = timeSecs*ephysData(iCount).ephys_sample_rate;


    plotRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);

    function KeyPressCb(~,evnt)
        fprintf('key pressed: %s\n',evnt.Key);
        if strcmpi(evnt.Key,'uparrow')
            timeChunkStart=timeChunkStop;
            timeChunkStop=timeChunkStop+timeChunk;
            
            clf;
            plotRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);
        elseif strcmpi(evnt.Key,'downarrow')
            timeChunkStop=timeChunkStart;
            timeChunkStart=timeChunkStart-timeChunk;
           
            clf;
            plotRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);
        elseif strcmpi(evnt.Key,'leftarrow')
            iCluster=iCluster+1;
            clf;
            plotRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);
        elseif strcmpi(evnt.Key,'rightarrow')
            iCluster=iCluster-1;
            clf;
            plotRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);
        end  
    end
 end
 
 %str templates 
 
 function plotRaw(memMapData, ephysData, iCluster, iCount ,timeChunkStart, timeChunkStop)
    %get the used channels
    chanAmps = squeeze(max(ephysData(iCount).templates(iCluster,:,:))-min(ephysData(iCount).templates(iCluster,:,:)));
    maxChan = find(chanAmps==max(chanAmps),1);
    maxXC = ephysData(iCount).channel_positions(maxChan,1); 
    maxYC = ephysData(iCount).channel_positions(maxChan,2);
    chanDistances = ((ephysData(iCount).channel_positions(:,1)-maxXC).^2 ...
        + (ephysData(iCount).channel_positions(:,2)-maxYC).^2).^0.5;
    chansToPlot = find(chanDistances<170);
    
    %get spike locations 
    pull_spikeT = -40:41;
    thisC=ephysData(iCount).spike_templates(iCluster);
    theseTimesCenter=ephysData(iCount).spike_times(ephysData(iCount).spike_templates==thisC);
    theseTimesCenter=theseTimesCenter(theseTimesCenter > timeChunkStart/ephysData(iCount).ephys_sample_rate);
    theseTimesCenter=theseTimesCenter(theseTimesCenter < timeChunkStop/ephysData(iCount).ephys_sample_rate);
    if ~isempty(theseTimesCenter)
        theseTimesFull = reshape(theseTimesCenter*ephysData(iCount).ephys_sample_rate+pull_spikeT, [size(theseTimesCenter,1)*size(pull_spikeT,2),1]);
    end
    %plot
%   cCount=cumsum(repmat(abs(max(max(memMapData(chansToPlot, timeChunkStart:timeChunkStop)))),size(chansToPlot,1),1),1);
    cCount=cumsum(repmat(1000,size(chansToPlot,1),1),1);
    

    t=timeChunkStart:timeChunkStop;
    LinePlotReducer(@plot, t, double(memMapData(chansToPlot, timeChunkStart:timeChunkStop))+double(cCount),'k');
    if ~isempty(theseTimesCenter)
        hold on;
        
        LinePlotReducer(@plot, theseTimesFull,double(memMapData(chansToPlot, theseTimesFull))+double(cCount),'r');
    end
    LinePlotExplorer(gcf);
    %overlay the spikes
 
 end