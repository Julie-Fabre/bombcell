function roundedNumber = prettify_roundUpNatural(number, decimalPlaces)

% sanitize inputs 
if nargin < 2 || isempty(decimalPlaces)
    decimalPlaces = 2;
end

% If the number is zero, return zero immediately
if number == 0
    roundedNumber = 0;
    return;
end

% Determine the number of places to move the first significant digit to the left of the decimal
shift = floor(log10(abs(number))) - 1;

% Shift the number so that the first significant digit is just to the left of the decimal
shiftedNumber = number / 10^shift;

% Round the shifted number up to n decimal places
roundedShiftedNumber = ceil(shiftedNumber * 1^decimalPlaces) / 1^decimalPlaces;

% Shift the number back to its original scale
roundedNumber = roundedShiftedNumber * 10^shift;


end
