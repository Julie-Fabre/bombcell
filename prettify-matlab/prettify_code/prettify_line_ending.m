function formatedLine = prettify_line_ending(lineToEnd, noPunctuationEnding, semiColonEnding)
    
if ~isempty(lineToEnd) && ~all(isspace(lineToEnd)) % if line is not empty
    if contains(lineToEnd(1:end), noPunctuationEnding) && ~endsWith(lineToEnd, '...')
        lineToEnd = regexprep(lineToEnd, '[,; \t]+$', ' ');
    else
        if semiColonEnding == 1
            if ~endsWith(lineToEnd, ';') && ~endsWith(lineToEnd, '...')
                % trim any spaces or other punctuation (commas)
                while endsWith(lineToEnd, {',', ' '})
                    lineToEnd = lineToEnd(1:end-1);
                end
                % add a semicolon
                lineToEnd = [lineToEnd ';'];
            end
        end
    end

    formatedLine = lineToEnd;
end

end