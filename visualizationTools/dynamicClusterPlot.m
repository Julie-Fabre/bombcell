 function dynamicClusterPlot(memMapData,ephysData,ephysParams,qMetrics)
    %set up dynamic figure
    h = figure; 
    set(h,'KeyPressFcn',@KeyPressCb) ;
    
    %initial conditions
    iCluster=1;
    iCount=1;
    timeSecs = 1;
    timeChunkStart = 5000;
    timeChunk = timeSecs*ephysData.ephys_sample_rate;
    timeChunkStop = timeSecs*ephysData.ephys_sample_rate;
    
    %plot initial conditions
    plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics);
    
    %change on keypress
    function KeyPressCb(~,evnt)
        fprintf('key pressed: %s\n',evnt.Key);
        if strcmpi(evnt.Key,'rightarrow')
            iCluster=iCluster+1;
            clf;
            plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics);
        elseif strcmpi(evnt.Key,'leftarrow')
            iCluster=iCluster-1;
            clf;
            plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics);
        elseif strcmpi(evnt.Key,'uparrow')
            timeChunkStart=timeChunkStop;
            timeChunkStop=timeChunkStop+timeChunk;
            
            clf;
            plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics);
        elseif strcmpi(evnt.Key,'downarrow')
            timeChunkStop=timeChunkStart;
            timeChunkStart=timeChunkStart-timeChunk;
           
            clf;
            plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics);
       
        end   
    end
 end

 %str templates 
 
 function plotFF(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop,ephysParams,qMetrics)
        %[strCluster,~]=find(ephysData(iCount).str_templates);
        qMetric(iCount).nSpikes(iCluster) = numel(ephysData(iCount).spike_times_timeline(ephysData(iCount).spike_templates == iCluster));
        FR=qMetric(iCount).nSpikes(iCluster)/(max(ephysData(iCount).spike_times_timeline)-min(ephysData(iCount).spike_times_timeline));
         
        theseSpikeTimes = ephysData(iCount).spike_times_timeline(ephysData(iCount).spike_templates == iCluster);
        [ccg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
        ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 1, 'norm', 'rate'); %function
    
        acgFR=ccg(:, 1, 1);
        ephysData(iCount).spike_times=ephysData(iCount).spike_times_timeline;
        ephysData(iCount).recordingDuration = (max(ephysData(iCount).spike_times_timeline)-min(ephysData(iCount).spike_times_timeline));
        subplot(5,13,[2:7, 15:20])
        cla;
        chanAmps = squeeze(max(ephysData(iCount).templates(iCluster,:,:))-min(ephysData(iCount).templates(iCluster,:,:)));
        maxChan = find(chanAmps==max(chanAmps),1);
        maxXC = ephysData(iCount).channel_positions(maxChan,1); 
        maxYC = ephysData(iCount).channel_positions(maxChan,2);
        chanDistances = ((ephysData(iCount).channel_positions(:,1)-maxXC).^2 ...
            + (ephysData(iCount).channel_positions(:,2)-maxYC).^2).^0.5;
        chansToPlot = find(chanDistances<170);
        for iChanToPlot=1:size(chansToPlot,1)
            plot((ephysData(iCount).waveform_t+(ephysData(iCount).channel_positions(chansToPlot(iChanToPlot),1)-11)/10),...
                squeeze(ephysData(iCount).templates(iCluster,:,chansToPlot(iChanToPlot)))'+ephysData(iCount).channel_positions(chansToPlot(iChanToPlot),2));

            hold on;
        end
        xlabel('Position(um)+Time(ms)');ylabel('Position(um)');title('Template waveform');
        
        
        subplot(5,13,[1,14,27,40,53]) % plot all template waveforms : ephysData(iCount).templates(iCluster,:,:)
        xlim([min(ephysData(iCount).channel_positions(:,1)),max(ephysData(iCount).channel_positions(:,1))]);
        ylim([min(ephysData(iCount).channel_positions(:,2)),max(ephysData(iCount).channel_positions(:,2))]);
        scatter(ephysData(iCount).channel_positions(:,1),ephysData(iCount).channel_positions(:,2),'filled')
        hold on;
        scatter(maxXC,maxYC,150,'filled');
        %striatum location lines 
%        str=ephysData(iCount).str_templates(1:size(ephysData(iCount).channel_positions,1));%QQ
%        scatter(ephysData(iCount).channel_positions(str,1),ephysData(iCount).channel_positions(str,2),'filled','g') 
        title('Location on probe')

        
        subplot(5,13,8:10)
        acgH=area( acgFR);xlim([500 1000]); %Check FR
        xlabel('time (ms)');
        hold on;
        %sup=find(acgFR(500:1000)>=ephysData(iCount).spike_rate(iCluster));
        %currAx=gca;
        %line([500+sup(1), 500+sup(1)], [currAx.YLim(1),currAx.YLim(2)]);
        
        mean2STDend = mean( acgFR(800:1000)) + 2*std( acgFR(800:1000));
        mean2STD = mean(acgFR(500:1000)) + 2*std(acgFR(500:1000));
        meanFR2STD = FR +  2*std(acgFR(800:1000));
        mean25STDend = mean( acgFR(800:1000)) + 2.5*std( acgFR(800:1000));
        mean25STD = mean(acgFR(500:1000)) + 2.5*std(acgFR(500:1000));
        meanFR25STD = FR +  2.5*std(acgFR(800:1000));
        mean3STDend = mean( acgFR(800:1000)) + 3*std( acgFR(800:1000));
        mean3STD = mean(acgFR(500:1000)) + 3*std(acgFR(500:1000));
        meanFR3STD = FR +  3*std(acgFR(800:1000));
        modeOverMean = mean(acgFR(530:560))/mean(acgFR(800:1000));
        ee=find(acgFR==max(acgFR))-500;
        thisC=ephysData(iCount).spike_templates(iCluster);
        theseTimes=ephysData(iCount).spike_times(ephysData(iCount).spike_templates==thisC);
        theseISI=diff(theseTimes);
        long_isi_total_2s =numel(find(theseISI > 2 ))/ephysData(iCount).recordingDuration;
       line([500,1000],[mean2STDend,mean2STDend]); text(800,mean2STDend, 'mean2STDend')
        line([500,1000],[mean2STD,mean2STD]); text(800,mean2STD, 'mean2STD')
        line([500,1000],[meanFR2STD,meanFR2STD]); text(800,meanFR2STD, 'meanFR2STD')
        
        line([500,1000],[mean25STDend,mean25STDend]); text(800,mean25STDend, 'mean25STDend')
        line([500,1000],[mean25STD,mean25STD]); text(800,mean25STD, 'mean25STD')
        line([500,1000],[meanFR25STD,meanFR25STD]); text(800,meanFR25STD, 'meanFR25STD')
        
        line([500,1000],[mean3STDend,mean3STDend]); text(800,mean3STDend, 'mean3STDend')
        line([500,1000],[mean3STD,mean3STD]); text(800,mean3STD, 'mean3STD')
        line([500,1000],[meanFR3STD,meanFR3STD]); text(800,meanFR3STD, 'meanFR3STD')
        
        
        ylabel('spikes per s')
        title('ACG');
        
%         subplot(5,13,11:13)
%         thisC=ephysData(iCount).spike_templates(iCluster);
%         theseTimes=ephysData(iCount).spike_times(ephysData(iCount).spike_templates==thisC);
%         theseISI=diff(theseTimes);
%         [sorted_isi, sorIdx] = sort(theseISI);
%         isiProba = histcounts(sorted_isi, [0:0.001:0.1], 'Normalization','probability');
%         plot(isiProba); %Check FR
%         long_isi_total_2s =numel(find(theseISI > 2 ))/ephysData(iCount).recordingDuration;
%         long_isi_total_5s = sum(theseISI(theseISI > 5 ))/numel(theseISI);
%         xlabel('time (ms)');title('ISI');
%         
%             
%         
%         ha=subplot(5,13,21:26);
%         pos = get(ha,'Position');   
%         un = get(ha,'Units');
%         delete(ha)
%         rowNames = {'Post-spike supression';'Spike rate';'Max';'PROP ISI>2s';...
%             'ContamRPV';'FractionRPV';'PercAmpliMissing'; 'NumACG';'Soma';'Waveform peaks';'cv2';...
%             'template duration'};
%         rowValues1 = [sup(1);ephysData(iCount).spike_rate(iCluster);ee(1); ...
%             long_isi_total_2s;qMetrics(iCount).contamRPV1(iCluster); ...
%             qMetrics(iCount).fractionRPV(iCluster);...
%             qMetrics(iCount).percMssgAmpli(iCluster);sum(acgFR(500:1000));...
%             qMetrics(iCount).somaCluster(iCluster); qMetrics(iCount).somaCluster(iCluster);
%             ephysParams(iCount).cv2(iCluster);...
%             qMetrics(iCount).duration(iCluster)];
%         rowValues2 = logical(zeros(size(rowValues1,1),1));
%         Tprop = table(rowValues1, rowValues2,'RowNames',rowNames);
%         t=uitable('Data',Tprop{:,:},...
%         'RowName',Tprop.Properties.RowNames,'Units', 'Normalized', 'Position',pos);
%         t.ColumnName = {'Value','Good'};
%         t.ColumnEditable = true;
%               
%     
%         
%         ephysData(iCount).mean2STDend(iCluster)=500-size(find(acgFR(1:500)<mean2STDend),2);
%         ephysData(iCount).mean2STD(iCluster)=500-size(find(acgFR(1:500)<mean2STD),2);
%         ephysData(iCount).meanFR2STD(iCluster)=500-size(find(acgFR(1:500)<meanFR2STD),2);
%         ephysData(iCount).mean25STDend(iCluster)=500-size(find(acgFR(1:500)<mean25STDend),2);
%         ephysData(iCount).mean25STD(iCluster)=500-size(find(acgFR(1:500)<mean25STD),2);
%         ephysData(iCount).meanFR25STD(iCluster)=500-size(find(acgFR(1:500)<meanFR25STD),2);
%         ephysData(iCount).mean3STDend(iCluster)=500-size(find(acgFR(1:500)<mean3STDend),2);
%         ephysData(iCount).mean3STD(iCluster)=500-size(find(acgFR(1:500)<mean3STD),2);
%         ephysData(iCount).meanFR3STD(iCluster)=500-size(find(acgFR(1:500)<meanFR3STD),2);
%         ephysData(iCount).modeACG(iCluster)=mode(acgFR(500:1000));
        
        
        
        subplot(5,13,[29:39])
        theseAmplis = ephysData(iCount).template_amplitudes(ephysData(iCount).spike_templates==thisC);
        theseTimes=theseTimes-theseTimes(1);%so doesn't start negative. QQ do alignement before 
        yyaxis left
        scatter(theseTimes, theseAmplis, 'blue','filled')
        hold on;
        currTimes=theseTimes(theseTimes>timeChunkStart/ephysData(iCount).recordingDuration & theseTimes<timeChunkStop/ephysData(iCount).recordingDuration );
        currAmplis=theseAmplis(theseTimes>timeChunkStart/ephysData(iCount).recordingDuration & theseTimes<timeChunkStop/ephysData(iCount).recordingDuration );
        scatter(currTimes,currAmplis,'black','filled');
        xlabel('Experiment time (s)');
        ylabel('Template amplitude scaling');
        axis tight
        hold on;
        ylim([0,round(max( theseAmplis))])
        set(gca,'YColor','b')
        yyaxis right
        binSize = 20;
        timeBins = 0:binSize:ceil(ephysData(iCount).spike_times(end));
        [n,x] = hist(theseTimes, timeBins);
        n = n./binSize;

        stairs(x,n, 'LineWidth', 2.0, 'Color', [1 0.5 0]);
        ylim([0,2*round(max(n))])
        set(gca,'YColor',[1 0.5 0])
        ylabel('Firing rate (sp/sec)');
        
        subplot(5,13,28)
        theseB=min(theseAmplis):0.5:max(theseAmplis);
        [counts,bins] = hist(theseAmplis,50); %# get counts and bin locations
        h=barh(qMetrics.ampliBinCenters{iCluster},qMetrics.ampliBinCounts{iCluster},'red');
        hold on;
        h.FaceAlpha = 0.5;
        plot(qMetrics.ampliFit{iCluster},qMetrics.ampliBinCenters{iCluster},'red')
        
%        line([0, max(counts)],[fitOutput(4), fitOutput(4)],'Color','red','LineWidth',2);
        
        subplot(5,13,41:46)
        title('Raw unwhitened data')
        hold on;
        plotSubRaw(memMapData,ephysData,iCluster,iCount, timeChunkStart, timeChunkStop);
        
          subplot(5,13,47:52)
                  pgood=   plot(ephysData(iCount).waveform_t , ...
        squeeze(ephysData(iCount).templates(iCluster, :, maxChan))*0.194');      
     
        [PKS,LOCS]=findpeaks(squeeze(ephysData(iCount).templates(iCluster, :,maxChan))*0.194,'MinPeakProminence',1);
        [TRS,LOCST]=findpeaks(squeeze(ephysData(iCount).templates(iCluster, :, maxChan))*-1*0.194,'MinPeakProminence',1);
        thesOnes=ones(size(pgood.XData));
        thesOnes([LOCS,LOCST])=0;
        pgood.XData(logical(thesOnes))=NaN;
        hold on;
        pgogggood=   plot(ephysData(iCount).waveform_t , ...
                squeeze(ephysData(iCount).templates(iCluster, :,maxChan))*0.194'); 

%pbb.YData(~[PKS,-TRS])=NaN;
    
    set(pgood, 'Marker', 'v');          
                    
    makepretty;
    
    %% to add: MH, isodistance, CCG with n most similar templates 
 end
 
 %str templates 
 
 function plotSubRaw(memMapData, ephysData, iCluster, iCount ,timeChunkStart, timeChunkStop)
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
    theseTimesCenter=theseTimesCenter(theseTimesCenter > timeChunkStart/ephysData.ephys_sample_rate);
    theseTimesCenter=theseTimesCenter(theseTimesCenter < timeChunkStop/ephysData.ephys_sample_rate);
    if ~isempty(theseTimesCenter)
        %theseTimesCenter=theseTimesCenter(1);
        theseTimesFull = theseTimesCenter*ephysData.ephys_sample_rate+pull_spikeT;
        %theseTimesFull=unique(sort(theseTimesFull));
    end
    %plot
%   cCount=cumsum(repmat(abs(max(max(memMapData(chansToPlot, timeChunkStart:timeChunkStop)))),size(chansToPlot,1),1),1);
    cCount=cumsum(repmat(1000,size(chansToPlot,1),1),1);
    

    t=timeChunkStart:timeChunkStop;
    LinePlotReducer(@plot, t, double(memMapData(chansToPlot, timeChunkStart:timeChunkStop))+double(cCount),'k');
    if ~isempty(theseTimesCenter)
        hold on;
        for iTimes=1:size(theseTimesCenter,1)
            if ~any(mod(theseTimesFull(iTimes,:),1))
            LinePlotReducer(@plot, theseTimesFull(iTimes,:),double(memMapData(chansToPlot, theseTimesFull(iTimes,:)))+double(cCount),'r');
        
            end
        end
    end
    LinePlotExplorer(gcf);
    %overlay the spikes
 
 end