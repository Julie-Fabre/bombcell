iUnit = iUnit+1;

%clf;%figure('Color', 'k'); 
subplot(1,5,[1:4])
hold on;
scatter(ephysData.spike_times_timeline(ephysData.spike_templates==iUnit), ephysData.template_amplitudes(ephysData.spike_templates==iUnit),...
    [],rgb('Red'), 'filled')
ylabel({'Spike-to-template'; 'scaling factor'},'Color','w', 'fontsize', 14)
xlabel('time (s)','Color','w', 'fontsize', 14)

set(gca,'XColor','w');
set(gca,'YColor','w');
%makepretty;
set(gca, 'Color','k')
YLIMS = ylim;
subplot(1,5,5)
timeChunks = [min(ephysData.spike_times_timeline(ephysData.spike_templates==iUnit)),...
    max(ephysData.spike_times_timeline(ephysData.spike_templates==iUnit))];
[percent_missing, bin_centers, num, n_fit_cut] = bc_percSpikesMissing(ephysData.template_amplitudes(ephysData.spike_templates==iUnit), ephysData.spike_times_timeline(ephysData.spike_templates==iUnit),...
    timeChunks, 0)
hold on;
b = barh(bin_centers, num);
b.FaceColor = rgb('Red');
xlabel('Count','Color','w', 'fontsize', 14)
set(gca, 'Color','k')
set(gca,'XColor','w');
set(gca,'YColor','w');
box off;
%makepretty;
set(gca, 'Color','k')
ylim([YLIMS(1), YLIMS(2)])
