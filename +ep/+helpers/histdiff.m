% HISTDIFF   Calculate histogram of differences.
%
%       [N, X] = HISTDIFF(Y1, Y2) returns a histogram with 10 equally
%       spaced bins of all the differences Y1(i) - Y2(j).  This is
%       useful for constructing correlograms of point process data,
%       stored as a list of times. BAR(X,N) plots the histogram.
%
%       HISTDIFF(Y1, Y2, N), where N is a scalar, uses N bins.
%
%       HISTDIFF(Y1, Y2, X), where X is a vector, calculates a histogram using
%       bins whose borders are specified in X -- there are
%       length(X)-1 bins.  NOTE: this behaviour is different to the
%       mathworks-supplied HIST.
%
%	HISTDIFF uses one of three algorithms, depending on the
%	inputs.  The slowest is general and makes no assumptions about
%	Y1 or Y2 and assumes only that the bin borders are
%	non-decreasing. If a number of bins is specified, or the
%	spacing between elements in X is regular, HISTDIFF exploits
%	this fact to speed up the calculation.  If, in addition, the
%	vectors Y1 and Y2 are sorted with the smallest value first, it
%	uses an even faster algorithm.
%
%       See also HIST, BAR.
