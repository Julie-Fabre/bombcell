%isdmatrix - Test if parameter is a matrix of doubles (>= 2 columns).
%
%  USAGE
%
%    test = isdmatrix(x,test1,test2,...)
%
%    x              parameter to test
%    test1...       optional list of additional tests
%
%  EXAMPLES
%
%    % Test if x is a matrix of doubles
%    isdmatrix(x)
%
%    % Test if x is a matrix of strictly positive doubles
%    isdmatrix(x,'>0')
%
%    % Special test: test if x is a 3-line matrix of doubles
%    isdmatrix(x,'#3')
%
%    % Special test: test if x is a 2-column matrix of doubles
%    isdmatrix(x,'@2')
%
%  NOTE
%
%    The tests ignore NaNs, e.g. isdmatrix([5e-3 nan;4 79]), isdmatrix([1.7 nan 3],'>0') and
%    isdmatrix([nan -7.4;nan nan;-2.3 -5],'<=0') all return 1.
%
%  SEE ALSO
%
%    See also isdvector, isdscalar, isimatrix, isivector, isiscalar, isastring,
%    islscalar, islvector, islmatrix.
%

% Copyright (C) 2010-2015 by Michaël Zugaro
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.

function test = isdmatrix(x,varargin)

% Check number of parameters
if nargin < 1,
  error('Incorrect number of parameters (type ''help <a href="matlab:help isdmatrix">isdmatrix</a>'' for details).');
end

% Test: doubles, two dimensions, two or more columns?
test = isa(x,'double') & length(size(x)) == 2 & size(x,2) >= 2;

% Optional tests
for i = 1:length(varargin),
	try
		if varargin{i}(1) == '#',
			if size(x,1) ~= str2num(varargin{i}(2:end)), test = false; return; end
		elseif varargin{i}(1) == '@',
			if size(x,2) ~= str2num(varargin{i}(2:end)), test = false; return; end
		elseif ~eval(['all(x(~isnan(x))' varargin{i} ');']), test = false; return; end
	catch err
		error(['Incorrect test ''' varargin{i} ''' (type ''help <a href="matlab:help isdmatrix">isdmatrix</a>'' for details).']);
	end
end
