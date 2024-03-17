function prettyCode = prettify_code(rawCode, xmlFile)
% format inputed code according to xmlFile rules 
% ------
% Inputs
% ------
% - rawCode: 1 x n char array of code to format
% - xmlFile: 1 x n char array containing the path to your xml formating file 
% -------
% Outputs
% -------
% - prettyCode: 1 x n char array of formated code 
% ------
% to do:
% - comments
% - spaces after parenthesis, commas ect 
% - commas
% - space around operators
% ------
% Julie M. J. Fabre

    % Parse the XML file
    xDoc = xmlread(xmlFile);
    
    % Get indentation settings
    indentNode = xDoc.getElementsByTagName('indent').item(0);
    spaceCount = str2double(indentNode.getElementsByTagName('spaceCount').item(0).getFirstChild.getData);
    increaseIndentation_java = split(indentNode.getElementsByTagName('increaseIndentation').item(0).getFirstChild.getData, ',');
    decreaseIndentation_java = split(indentNode.getElementsByTagName('decreaseIndentation').item(0).getFirstChild.getData, ',');
    increaseIndentation = arrayfun(@(x)  increaseIndentation_java(x).toCharArray', 1:size(increaseIndentation_java,1), 'UniformOutput', false);
    decreaseIndentation = arrayfun(@(x)  decreaseIndentation_java(x).toCharArray', 1:size(decreaseIndentation_java,1), 'UniformOutput', false);
    increaseIndentation_single_java = split(indentNode.getElementsByTagName('singleIncreaseIndentation').item(0).getFirstChild.getData, ',');
    increaseIndentation_single = arrayfun(@(x)  increaseIndentation_single_java(x).toCharArray', 1:size(increaseIndentation_single_java,1), 'UniformOutput', false);
    
    % Get spacing settings
    spacingNode = xDoc.getElementsByTagName('spacing').item(0);
    aroundOperators = strcmp(spacingNode.getElementsByTagName('aroundOperators').item(0).getFirstChild.getData, 'true');
    afterCommentOperator = strcmp(spacingNode.getElementsByTagName('afterCommentOperator').item(0).getFirstChild.getData, 'true');
    afterComma = strcmp(spacingNode.getElementsByTagName('afterComma').item(0).getFirstChild.getData, 'true');
    
    % Get blank lines settings
    blankLinesNode = xDoc.getElementsByTagName('blankLines').item(0);
    addBlankLines = strcmp(blankLinesNode.getElementsByTagName('addBlankLines').item(0).getFirstChild.getData, 'true');
    if addBlankLines
        afterKeywords_java = split(blankLinesNode.getElementsByTagName('afterKeywords').item(0).getFirstChild.getData, ',');
        beforeKeywords_java = split(blankLinesNode.getElementsByTagName('beforeKeywords').item(0).getFirstChild.getData, ',');
        afterKeywords = arrayfun(@(x)  afterKeywords_java(x).toCharArray', 1:size(afterKeywords_java,1), 'UniformOutput', false);
        beforeKeywords = arrayfun(@(x)  beforeKeywords_java(x).toCharArray', 1:size(beforeKeywords_java,1), 'UniformOutput', false);
    end
    singleBlankLines = strcmp(blankLinesNode.getElementsByTagName('singleBlankLines').item(0).getFirstChild.getData, 'true');

    % Get newline settings
    singleLinesNode = xDoc.getElementsByTagName('singleLines').item(0);
    newLineBeforeKeywords_java = split(singleLinesNode.getElementsByTagName('newLineBefore').item(0).getFirstChild.getData, ',');
    newLineBeforeKeywords = arrayfun(@(x)  newLineBeforeKeywords_java(x).toCharArray', 1:size(newLineBeforeKeywords_java,1), 'UniformOutput', false);
    newLineAfterKeywords_java = split(singleLinesNode.getElementsByTagName('newLineAfter').item(0).getFirstChild.getData, ',');
    newLineAfterKeywords = arrayfun(@(x)  newLineAfterKeywords_java(x).toCharArray', 1:size(newLineAfterKeywords_java,1), 'UniformOutput', false);

    % Get line ending settings
    lineEndingNode = xDoc.getElementsByTagName('lineEndings').item(0);
    noPunctuationEnding_java = split(lineEndingNode.getElementsByTagName('noPunctuationEnding').item(0).getFirstChild.getData, ',');
    noPunctuationEnding = arrayfun(@(x)  noPunctuationEnding_java(x).toCharArray', 1:size(noPunctuationEnding_java,1), 'UniformOutput', false);
    semicolonEnding = strcmp(lineEndingNode.getElementsByTagName('semicolonEnding').item(0).getFirstChild.getData, 'true');

    % Apply beautification
    codeLines = split(rawCode, newline);
    indentLevel = 0;
    iLine = 1;
    single_indent = false;
    functioncomments = false;

    while iLine <= numel(codeLines)
        
        % get line 
        thisLine = strtrim(codeLines{iLine});
        
        % remove any leading or trailing white space
        thisLine = prettify_trim_white_space(thisLine);

        % If line is a comment, skip beautification and move to next line
        if startsWith(thisLine, '%')
            commentLine = true;
            % Extracting comment and ensuring the format is "% "
            rawComment = strtrim(thisLine);
            thisLine = prettify_comments(rawComment);
        else
            commentLine = false;
            % Detect if there's an inline comment
            commentIdx = strfind(thisLine, '%');
            
            if ~isempty(commentIdx)
                % If there's a comment, split the line into code and comment
                thisLine = strtrim(thisLine(1:commentIdx(1)-1));
                commentPart = prettify_comments(thisLine(commentIdx(1):end));
            else
                thisLine = thisLine;
                commentPart = '';
            end

            % split string if contains relevant keywords 
            [codeLines, thisLine] = prettify_split_lines(codeLines, thisLine, newLineBeforeKeywords, newLineAfterKeywords, iLine);

            thisLine = prettify_trim_white_space(thisLine);
    
            % add indent after keywords 
            if any(endsWith(thisLine, decreaseIndentation)) || any(startsWith(thisLine, decreaseIndentation))
                indentLevel = indentLevel - 1;
            end
            
            thisLine = regexprep(thisLine, '\s{2,}', ' '); % remove any double (or more) spaces

            % make the line ending pretty (add/remove commas and semicolons
            % as necessary)
            thisLine = prettify_line_ending(thisLine, noPunctuationEnding, semicolonEnding);

            if ~isempty(commentPart)
                thisLine = [thisLine, ' ', commentPart];
            else
                thisLine = thisLine;
            end

            
        end
        codeLines{iLine} = [repmat(' ', 1, spaceCount * indentLevel),  thisLine]; % store line
        
        if ~commentLine
            % remove indent for next line if it's of type single line 
            if single_indent
                indentLevel = indentLevel - 1;
                single_indent = false;
            end
    
            % add indent if keyword
            if any(startsWith(thisLine, increaseIndentation))
                indentLevel = indentLevel + 1;
            end
            
            % remove indent if it was a single line indent 
            if any(endsWith(thisLine, increaseIndentation_single))
                indentLevel = indentLevel + 1;
                single_indent = true;
            end
        end

        if addBlankLines
            % Add blank lines after specific keywords
            if any(startsWith(line, afterKeywords)) && (iLine == numel(lines) || ~isempty(lines{iLine+1}))
                lines = [lines(1:iLine); ""; lines(iLine+1:end)];
                iLine = iLine + 1;
            end
    
            % Add blank lines before specific keywords
            if any(startsWith(line, beforeKeywords)) && (iLine == 1 || ~isempty(lines{iLine-1}))
                lines = [lines(1:iLine-1); ""; lines(iLine:end)];
                iLine = iLine + 1;
        end
        end
        
        iLine = iLine + 1;
    end

    % Remove surplus blank lines
    codeLines = regexprep(codeLines, '^\s*$', '');
    codeLines(cellfun(@isempty, codeLines) & [true; cellfun(@isempty, codeLines(1:end-1))]) = [];
    
    prettyCode = strjoin(codeLines, newline);

    if aroundOperators % add a space around operators, if there isn't one already
        operatorsPattern = '(?<!\s)(=|<|>|~|&|\||-|\+|\*|/|\^)';
        specialCases = {'& &', '| |', '= =', '~ =', '\. /', '\. \', '. ^', '&  &', '|  |', '=  =', '~  =', '\.  /', '\.  \', '\.  ^'};
        specialCases_replace = {'&&', '||', '==', '~=', '\./', '\.\', '.^','&&', '||', '==', '~=', '\./', '\.\', '.^'};
        prettyCode = regexprep(prettyCode, operatorsPattern, ' $1');
        for iLine = 1:length(specialCases)
            prettyCode = strrep(prettyCode, specialCases{iLine}, specialCases_replace{iLine});
        end
        operatorsPattern = '(=|<|>|~|&|\||-|\+|\*|/|\^)(?!\s)';
        specialCases = {'& &', '| |', '= =', '~ =', '\. /', '\. \', '\. ^', '&  &', '|  |', '=  =', '~  =', '\.  /', '\.  \', '.  ^'};
        specialCases_replace = {'&&', '||', '==', '~=', '\./', '\.\', '\.^','&&', '||', '==', '~=', '\./', '\.\', '\.^'};
        prettyCode = regexprep(prettyCode, operatorsPattern, '$1 ');
        for iLine = 1:length(specialCases)
            prettyCode = strrep(prettyCode, specialCases{iLine}, specialCases_replace{iLine});
        end
    end

    if afterCommentOperator % add a space after comment operators, if there isn't one already
       commentOperatorPattern = '(%)(?!\s)';
       specialCases = {'% %','%   %'};
       specialCases_replace = {'%%', '%%'};
       prettyCode = regexprep(prettyCode, commentOperatorPattern, '$1 ');
       for iLine = 1:length(specialCases)
           prettyCode = strrep(prettyCode, specialCases{iLine}, specialCases_replace{iLine});
       end
    end

    % Remove spaces around ( and )
    prettyCode = regexprep(prettyCode, '[ \t]*\([ \t]*', '(');
    prettyCode = regexprep(prettyCode, '[ \t]*\)[ \t]*', ')');
    
    if afterComma % add a space after commas, if there isn't one already
        prettyCode = regexprep(prettyCode, '(?<!\s)(,)(?!\s)', ', ');
    end

    if singleBlankLines % remove any double (or more) blank lines
        prettyCode = regexprep(prettyCode, '^(\s*\r\n){2,}', '\r\n');
    end

end
