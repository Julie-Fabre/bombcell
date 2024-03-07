% Sample data for two groups
group1 = randn(10, 1) + 1; % Random data centered around 1
group2 = randn(10, 1) + 1; % Random data centered around 2
group3 = randn(10, 1);
group4 = randn(10, 1) + 5;
group5 = randn(10, 1) + 5;

figure;
bar_data = [mean(group1), mean(group2), mean(group3), mean(group4), mean(group5)]; % Use mean values for the bar heights
bar(bar_data);
set(gca, 'XTickLabel', {'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'});
ylabel('Value');
title('Bar plot with p-value annotations');
% Perform a t-test to get the p-value
[~, pval(1)] = ttest2(group1, group2);
[~, pval(2)] = ttest2(group1, group3);
[~, pval(3)] = ttest2(group1, group4);
[~, pval(4)] = ttest2(group1, group5);
[~, pval(5)] = ttest2(group2, group3);
[~, pval(6)] = ttest2(group2, group4);
[~, pval(7)] = ttest2(group2, group5);
[~, pval(8)] = ttest2(group3, group4);
[~, pval(9)] = ttest2(group3, group5);
[~, pval(10)] = ttest2(group4, group5);
% Assuming we're working with the current axes
ax = gca;
% Add the p-value to the plot
% Here we assume that the p-value should appear at y = 3, change this value as needed for your plot.
% x-coordinates (1 and 2) correspond to the bar centers.
prettify_pvalues(ax, [1,1,1,1,2,2,2,3,3,4], [2,3,4,5,3,4,5,4,5,5], pval,'PlotNonSignif', false);
prettify_plot('FigureColor', 'k');