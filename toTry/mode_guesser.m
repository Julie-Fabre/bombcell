function m = mode_guesser(x,p)
% UltraMegaSort2000 by Hill DN, Mehta SB, & Kleinfeld D  - 07/09/2010
%
% mode_guesser - guess mode of the 
%
% Usage:
%    m = mode_guesser(x,p)
%
% Description:
%   Guesses mode by looking for location where the data is most tightly
% distributed.  This is accomplished by sorting the vector x and 
% looking for the p*100 percentile range of data with the least range.
%
% Input: 
%   x - [1 x M] vector of scalars
%
% Option input:
%   p - proportion of data to use in guessing the mode, defaults to 0.1
%
% Output:
%   m - guessed value of mode
%

    %check for whether p is specified
    if nargin < 2, p = .1; end

    % determine how many samples is p proportion of the data
    num_samples = length(x);
    shift = round( num_samples * p );
    
    % find the range of the most tightly distributed data
    x = sort(x);
    [val,m_spot] = min( x(shift+1:end) - x(1:end-shift) );
    
    % use median of the tightest range as the guess of the mode
    m = x( round(m_spot + (shift/2)) );