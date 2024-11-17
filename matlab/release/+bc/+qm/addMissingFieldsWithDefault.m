function [data, missingFields] = addMissingFieldsWithDefault(data, defaultValues)
% JF, Check input structure has all necessary fields + add them with
% defualt values if not. 
% ------
% Inputs
% ------
    if ~isstruct(data) && ~istable(data)
        error('Input must be a structure or table');
    end
    
    if ~isstruct(defaultValues)
        error('Default values must be provided as a structure');
    end
    
    fieldnames = fields(defaultValues);
    
    if isstruct(data)
        missingFields = fieldnames(~isfield(data, fieldnames));
        
        for i = 1:length(missingFields)
            fieldName = missingFields{i};
            data.(fieldName) = defaultValues.(fieldName);
        end
    else  % data is a table
        existingFields = data.Properties.VariableNames;
        missingFields = setdiff(fieldnames, existingFields);
        
        for i = 1:length(missingFields)
            fieldName = missingFields{i};
            data.(fieldName) = repmat(defaultValues.(fieldName), height(data), 1);
        end
    end
end

