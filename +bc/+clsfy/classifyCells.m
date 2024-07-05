function unitClassif = classifyCells(ephysProperties, paramEP, region)
% classify striatal, GPe and cortical cells
unitClassif = cell(size(ephysProperties,1),1);


if ismember(region, {'CP', 'STR', 'Striatum', 'DMS', 'DLS', 'PS'}) %striatum, classification as in Peters et al., Nature, 2021
    unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_CP_threshold &...
        ephysProperties.postSpikeSuppression_ms < paramEP.postSpikeSup_CP_threshold) = {'MSN'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_CP_threshold &...
        ephysProperties.propLongISI <= paramEP.propISI_CP_threshold) = {'FSI'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_CP_threshold &...
        ephysProperties.postSpikeSuppression_ms >= paramEP.postSpikeSup_CP_threshold) = {'TAN'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_CP_threshold &...
        ephysProperties.propLongISI > paramEP.propISI_CP_threshold) = {'UIN'};

    figure();
    scatter3(ephysProperties.waveformDuration_peakTrough_us(strcmp(unitClassif, 'MSN')),...
        ephysProperties.postSpikeSuppression_ms(strcmp(unitClassif, 'MSN')), ephysProperties.propLongISI(strcmp(unitClassif, 'MSN')),...
        4, 'filled'); hold on;
    scatter3(ephysProperties.waveformDuration_peakTrough_us(strcmp(unitClassif, 'FSI')),...
        ephysProperties.postSpikeSuppression_ms(strcmp(unitClassif, 'FSI')), ephysProperties.propLongISI(strcmp(unitClassif, 'FSI')),...
        4, 'filled');
    scatter3(ephysProperties.waveformDuration_peakTrough_us(strcmp(unitClassif, 'TAN')),...
        ephysProperties.postSpikeSuppression_ms(strcmp(unitClassif, 'TAN')), ephysProperties.propLongISI(strcmp(unitClassif, 'TAN')),...
        4, 'filled');
    scatter3(ephysProperties.waveformDuration_peakTrough_us(strcmp(unitClassif, 'UIN')),...
        ephysProperties.postSpikeSuppression_ms(strcmp(unitClassif, 'UIN')), ephysProperties.propLongISI(strcmp(unitClassif, 'UIN')),...
        4, 'filled');
    set(gca, 'YDir', 'reverse' );
    xlabel('waveform duration (us)');
    ylabel('post spike suppression (ms)');
    zlabel('frac. ISI > 2s');
    legend({'MSN', 'FSI', 'TAN', 'UIN'})
    prettify_plot();

elseif ismember(region, {'Ctx', 'Cortex', 'Cortical'}) % cortex, classification as in Peters et al., Cell Reports, 2022
    unitClassif(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_Ctx_threshold) = {'Wide-spiking'};

    unitClassif(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_Ctx_threshold) = {'Narrow-spiking'};

    figure();
    histogram(ephysProperties.waveformDuration_peakTrough_us(ephysProperties.waveformDuration_peakTrough_us > paramEP.templateDuration_Ctx_threshold),...
        'BinEdges',[0:10:1000]); hold on;
    histogram(ephysProperties.waveformDuration_peakTrough_us(ephysProperties.waveformDuration_peakTrough_us <= paramEP.templateDuration_Ctx_threshold),...
        'BinEdges',[0:10:1000]);
    xlabel('waveform duration (us)');
    ylabel('# of cells');
    legend({'Wide-spiking', 'Narrow-spiking'})
    prettify_plot();


% elseif ismember(region, {'GPe', 'Globus Pallidus external'}) % GPe - work
% in progress  
end

end
