function [structure, missingFields] = bc_addMissingFieldsWithDefault(structure, defaultValues)
% JF, Check input structure has all necessary fields + add them with
% defualt values if not. 
% ------
% Inputs
% ------
    if ~isstruct(structure)
        error('Input must be a structure');
    end
    
    if ~isstruct(defaultValues)
        error('Default values must be provided as a structure');
    end
    
    fieldnames = fields(defaultValues);
    missingFields = fieldnames(~isfield(structure, fieldnames));
    
    for i = 1:length(missingFields)
        fieldName = missingFields{i};
        structure.(fieldName) = defaultValues.(fieldName);
    end
end
