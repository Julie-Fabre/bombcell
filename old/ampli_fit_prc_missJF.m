% template amplitude
function [percent_missing_ndtr, fitOutput] = ampli_fit_prc_missJF(curr_amplis, toPlot)
% JF 20191114

%for iTemplate=unique(spike_templates)'
coords = curr_amplis;
nBins = 50;
[num, bins] = histcounts(curr_amplis, nBins);

%fit a gaussian to the histogram
mode_seed = bins(num == max(num)); %mode of mean_seed
bin_steps = diff(bins(1:2)); %size of a bin
x = bins(1:end-1) + bin_steps / 2; %x=value of the center of each bin
next_low_bin = x(1) - bin_steps;
add_points = next_low_bin:-bin_steps:0;%add points so amplitude values starts at 0
x = [add_points, x];
num = [zeros(size(add_points, 2), 1)', num];

if numel(mode_seed) > 1
    mode_seed = mean(mode_seed);
end
p0 = [max(num), mode_seed, 2 * nanstd(coords), prctile(curr_amplis, 1)];

% %%crap
% myfittype = fittype(@(a, x0, sigma,x) (a.*exp(-(x-x0).^2/(2*sigma^2))) ,...
%         'coefficients',...
%         {'a','x0','sigma'}, 'dependent',{'y'}, 'independent',{'x'}); %gaussian fit type

%give up and call python, where life makes sense and functions are pretty
echo off;
%lsqcurvefit(fun,x0,xdata,ydata)

%pyenv('Version','/usr/bin/python3.8')
py.importlib.import_module('gaussFitJF');
warning('off','all')
py.importlib.import_module('numpy');
p0_py = py.numpy.array(p0);
x_py = py.numpy.array(x);
num_py = py.numpy.array(num);
py.gaussFitJF.JF_fit(x_py, num_py, p0_py);
fitOutput = cell(ans);
fitOutput = np2mat2(fitOutput{1});

n_fit = JF_gaussian_cut(x, fitOutput(1), fitOutput(2), fitOutput(3), fitOutput(4));
min_amplitude = fitOutput(4);

%norm area calculated by fit parameters
norm_area_ndtr = normcdf((fitOutput(2) - min_amplitude)/fitOutput(3)); %ndtr((popt[1] - min_amplitude) /popt[2])
percent_missing_ndtr = 100 * (1 - norm_area_ndtr);

if toPlot == 1 % plot the amplitudes %%QQ
    %histogram
    figure();
    barh(histcounts(curr_amplis, 50));
    hold on;
    n_fit_no_cut = JF_gaussian_cut(x, fitOutput(1), fitOutput(2), fitOutput(3), 0);
    plot(n_fit_no_cut, x, 'r');
    hold on;
    %maxs=max(n_fit);
    plot(n_fit, x, 'b')
end
end
