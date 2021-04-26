cellTypes = {'MSN', 'FSI', 'TAN', 'UIN'};
celltype_col = ...
    [0.9,0.4,0.6; ... %msn
    0.4,0.5,0.6; ...%fsi
    0.5,0.3,0.1; ... %tan
    1,0.5,0];%iun
waveform_t = 1e3 * ((0:size(ephysData.templates, 2) - 1) / ephysData.ephys_sample_rate);
figure();

for iCellType = 1:size(cellTypes,2)
    subplot(2,4,iCellType)
    %waveforms
    P(iCellType)=plot(waveform_t, mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells,:)),':','Color',celltype_col(iCellType,:),'LineWidth',2);
    plotshaded(waveform_t,[-std(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells,:)) + mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells,:)); ...
    std(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells,:)) + mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells,:))],celltype_col(iCellType,:));
       
   
    
    subplot(2,4,iCellType+size(cellTypes,2))
    %ACGs
    ephysParams.ACG
end