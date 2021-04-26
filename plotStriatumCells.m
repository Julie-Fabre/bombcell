cellTypes = {'MSN', 'FSI', 'TAN', 'UIN'};
celltype_col = ...
    [0.9, 0.4, 0.6; ... %msn
    0.4, 0.5, 0.6; ... %fsi
    0.5, 0.3, 0.1; ... %tan
    1, 0.5, 0]; %iun
waveform_t = 1e3 * ((0:size(ephysData.templates, 2) - 1) / ephysData.ephys_sample_rate);
figure();

for iCellType = 1:size(cellTypes, 2)
    subplot(2, 4, iCellType)
    %waveforms
    P(iCellType) = plot(waveform_t, mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells, :)),  'Color', celltype_col(iCellType, :), 'LineWidth', 2);
    plotshaded(waveform_t, [-std(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells, :)) + mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells, :)); ...
        std(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells, :)) + mean(qMetric.waveformRaw(goodUnits & cellTypesClassif(iCellType).cells, :))], celltype_col(iCellType, :));
    xlim([0, waveform_t(end)])
    xlabel('time (ms)')
    ylabel('\mu V')
    makepretty;


    subplot(2, 4, iCellType+size(cellTypes, 2))
    %ACGs
    plot(0:0.001:0.5, nanmean(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end)), 'Color',  celltype_col(iCellType, :))
    %plot(0:0.001:0.5,mean(acg(fsi,500:end)),'b');%,'FaceColor','b');
    hold on;
    plotshaded(0:0.001:0.5, [-nanstd(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end)) + nanmean(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end)); ...
        nanstd(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end)) + nanmean(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end))], celltype_col(iCellType, :));
    area(0:0.001:0.5, nanmean(ephysParams.ACG(goodUnits & cellTypesClassif(iCellType).cells,  501:end)), 'FaceColor',  celltype_col(iCellType, :));
    xlim([0, 0.5])
    ylim([0, 50])
    ylabel('sp/s')
    xlabel('Time (s)')
    makepretty;
end