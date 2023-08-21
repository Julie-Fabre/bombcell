function prettyCode = makeCodePretty(rawCode, xmlFile)

    % Parse the XML file
    xDoc = xmlread(xmlFile);
    
    % Get indentation settings
    indentNode = xDoc.getElementsByTagName('indent').item(0);
    spaceCount = str2double(indentNode.getElementsByTagName('spaceCount').item(0).getFirstChild.getData);
    increaseIndentation_java = split(indentNode.getElementsByTagName('increaseIndentation').item(0).getFirstChild.getData, ',');
    decreaseIndentation_java = split(indentNode.getElementsByTagName('decreaseIndentation').item(0).getFirstChild.getData, ',');
    increaseIndentation = arrayfun(@(x)  increaseIndentation_java(x).toCharArray', 1:size(increaseIndentation_java,1), 'UniformOutput', false);
    decreaseIndentation = arrayfun(@(x)  decreaseIndentation_java(x).toCharArray', 1:size(decreaseIndentation_java,1), 'UniformOutput', false);
    
    % Get spacing settings
    spacingNode = xDoc.getElementsByTagName('spacing').item(0);
    aroundOperators = strcmp(spacingNode.getElementsByTagName('aroundOperators').item(0).getFirstChild.getData, 'true');
    afterComma = strcmp(spacingNode.getElementsByTagName('afterComma').item(0).getFirstChild.getData, 'true');
    
    % Apply beautification
    lines = split(rawCode, newline);
    indentLevel = 0;
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        
        if any(startsWith(line, decreaseIndentation)) || any(endsWith(line, increaseIndentation))
            indentLevel = max(0, indentLevel - 1);
        end
        
        lines{i} = [repmat(' ', 1, spaceCount * indentLevel),  line];
        
        if any(startsWith(line, increaseIndentation))
            indentLevel = indentLevel + 1;
        end
    end

    prettyCode = strjoin(lines, newline);

    if aroundOperators
        prettyCode = regexprep(prettyCode, '(=|<|>|~|&|\||-|\+|\*|/|\^)', ' $1 ');
    end

    if afterComma
        prettyCode = regexprep(prettyCode, ',', ', ');
    end

    prettyCode = regexprep(prettyCode, '[^\S\n]{2,}', ' ');
    prettyCode = regexprep(prettyCode, '[ \t]+$', '');
end
