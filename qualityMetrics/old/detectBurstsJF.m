function [burstStarts, burstStops, numberBurstSpikes] = detectBurstsJF(theseSpikes)  % burstSt = diff(theseSpikes)<0.15 & diff(theseSpikes)>0.05;%less than 1.5ms and more than 0.05ms 
  % consecDiff = burstSt(1:end-1)==1 & diff(burstSt)==1;
  theseisi=diff(theseSpikes);
   burstStops = find(diff(theseSpikes(1:end-1))<0.15 & diff(theseSpikes(1:end-1))>0.05 & ...
       diff(theseSpikes(2:end)) > 0.28);%when burst stops:isi before : less than 1.5ms and more than 0.05ms,
   %this uisi > 0.280 ms %like Jin Costa 2008
   if isempty(burstStops)
       numberBurstSpikes = NaN;
       burstStarts=NaN;
       burstStops=NaN;
   else
   for iBurstSp = 1:numel(burstStops)
       if iBurstSp==1
           theseB=find(theseisi(1:burstStops(iBurstSp))<0.15 & theseisi(1:burstStops(iBurstSp))>0.05);
           theseBc=(diff(theseB)==1);
           if ~isempty(find(theseBc==0, 1, 'last')) && theseisi(burstStops(iBurstSp)-1)<0.15 &&...
                   theseisi(burstStops(iBurstSp)-1)>0.05  %before last ok=counted, take last Bc, last 0  
               numberBurstSpikes(iBurstSp) = numel(theseBc)-find(theseBc==0, 1, 'last');
               burstStarts(iBurstSp)=burstStops(iBurstSp)-numberBurstSpikes(iBurstSp);
           else
               numberBurstSpikes(iBurstSp) = 1;
               burstStarts(iBurstSp)=burstStops(iBurstSp)-1;
           end
       else
           theseB=find(theseisi(burstStops(iBurstSp-1):burstStops(iBurstSp))<0.15 & theseisi(burstStops(iBurstSp-1):burstStops(iBurstSp))>0.05);
           theseBc=(diff(theseB)==1);
           if ~isempty(theseBc) && theseisi(burstStops(iBurstSp)-1)<0.15 &&...
                   theseisi(burstStops(iBurstSp)-1)>0.05  %before last ok=counted, take last Bc, last 0  
               numberBurstSpikes(iBurstSp) = numel(theseBc)-find(theseBc==0, 1, 'last');
               burstStarts(iBurstSp)=burstStops(iBurstSp)-numberBurstSpikes(iBurstSp);
           else
               numberBurstSpikes(iBurstSp) = 1;
               burstStarts(iBurstSp)=burstStops(iBurstSp)-1;
           end
       end
   end
   end
   %QQ later, grab these spikes from CAR raw data to plot waveforms. 
end