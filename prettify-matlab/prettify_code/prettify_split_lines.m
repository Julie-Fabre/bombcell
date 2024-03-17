function [codeLines, splitLine] = prettify_split_lines(codeLines, linesToSplit, newLineBeforeKeywords, newLineAfterKeywords, iLine)
    
    if contains(linesToSplit(2:end), newLineBeforeKeywords)
        for iNewLine = 1:size(newLineBeforeKeywords,2) 
            thiskey = newLineBeforeKeywords{iNewLine};
            % all possible combinations 
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ' '], [newline newLineBeforeKeywords{iNewLine} ' ']);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ' '], [',' newline newLineBeforeKeywords{iNewLine} ' ']);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ','], [',' newline newLineBeforeKeywords{iNewLine} ' ']);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ','], [newline newLineBeforeKeywords{iNewLine} ',']);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey '('], [newline newLineBeforeKeywords{iNewLine} '(']);
            linesToSplit = regexprep(linesToSplit, [',' thiskey '('], [',' newline newLineBeforeKeywords{iNewLine} '(']);
            linesToSplit = regexprep(linesToSplit, [';' thiskey '('], [';' newline newLineBeforeKeywords{iNewLine} '(']);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ','], [';' newline newLineBeforeKeywords{iNewLine} ',']);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ' '], [';' newline newLineBeforeKeywords{iNewLine} ' ']);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ';'], [';' newline newLineBeforeKeywords{iNewLine} ';']);
            linesToSplit = regexprep(linesToSplit, [')' thiskey ';'], [')' newline newLineBeforeKeywords{iNewLine} ';']);
            linesToSplit = regexprep(linesToSplit, [')' thiskey ' '], [')' newline newLineBeforeKeywords{iNewLine} ' ']);
            linesToSplit = regexprep(linesToSplit, [')' thiskey '('], [')' newline newLineBeforeKeywords{iNewLine} '(']);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ';'], [',' newline newLineBeforeKeywords{iNewLine} ';']);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ';'], [newline newLineBeforeKeywords{iNewLine} ';']);
            if endsWith(linesToSplit, newLineBeforeKeywords{iNewLine}) 
                 linesToSplit = regexprep(linesToSplit, newLineBeforeKeywords{iNewLine}, [newline newLineBeforeKeywords{iNewLine}]);
            end
        end

       linesToSplit = split(linesToSplit, newline);
       
       if size(linesToSplit,1) > 1
           linesToSplit = linesToSplit(arrayfun(@(x) ~isempty(linesToSplit{x}), 1:size(linesToSplit,1)));
            for iNewLine = 1:size(linesToSplit,1)
                if iNewLine == 1
                    codeLines = [codeLines(1:iLine+iNewLine-1-1); linesToSplit{iNewLine}; codeLines(iLine+iNewLine:end)];
                else
                    codeLines = [codeLines(1:iLine+iNewLine-1-1); linesToSplit{iNewLine}; codeLines(iLine+iNewLine-1:end)];
                end
            end
       end
       linesToSplit = linesToSplit{1};
    end

    if contains(linesToSplit(1:end), newLineAfterKeywords)
        for iNewLine = 1:size(newLineAfterKeywords,2)
            thiskey = newLineAfterKeywords{iNewLine};
            % all possible combinations
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ' '], [' ' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ' '], [' ' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ' '], [',' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ','], [',' newLineAfterKeywords{iNewLine} ',' newline]);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ','], [' ' newLineAfterKeywords{iNewLine} ',' newline]);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey '('], [' ' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [',' thiskey '('], [',' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [';' thiskey '('], [' ' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ','], [' ' newLineAfterKeywords{iNewLine} ',' newline]);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ' '], [' ' newLineAfterKeywords{iNewLine} newline]);
            linesToSplit = regexprep(linesToSplit, [';' thiskey ';'], [' ' newLineAfterKeywords{iNewLine} ';' newline]);
            linesToSplit = regexprep(linesToSplit, [')' thiskey ';'], [')' newLineAfterKeywords{iNewLine} ';' newline]);
            linesToSplit = regexprep(linesToSplit, [')' thiskey ' '], [')' newLineAfterKeywords{iNewLine} ' ' newline]);
            linesToSplit = regexprep(linesToSplit, [',' thiskey ';'], [',' newLineAfterKeywords{iNewLine} ';' newline]);
            linesToSplit = regexprep(linesToSplit, [' ' thiskey ';'], [' ' newLineAfterKeywords{iNewLine} ';' newline]);

            if endsWith(linesToSplit, newLineAfterKeywords{iNewLine})
                linesToSplit = regexprep(linesToSplit, newLineAfterKeywords{iNewLine}, [newLineAfterKeywords{iNewLine} newline]);
            elseif startsWith(linesToSplit, newLineAfterKeywords{iNewLine}) &&...
                    length(linesToSplit(~isspace(linesToSplit))) > length(newLineAfterKeywords{iNewLine}) + 1
                linesToSplit = regexprep(linesToSplit, [thiskey ';'], [newLineAfterKeywords{iNewLine} ';' newline]);
                linesToSplit = regexprep(linesToSplit, [thiskey ','], [newLineAfterKeywords{iNewLine} ',' newline]);
                linesToSplit = regexprep(linesToSplit, [thiskey ' '], [newLineAfterKeywords{iNewLine} ' ' newline]);
                linesToSplit = regexprep(linesToSplit, [thiskey '('], [newLineAfterKeywords{iNewLine} newline '(' ]);
            end
        end
       linesToSplit = split(linesToSplit, newline);
      
       if size(linesToSplit,1) > 1
           linesToSplit = linesToSplit(arrayfun(@(x) ~isempty(linesToSplit{x}), 1:size(linesToSplit,1)));
           for iNewLine = 1:size(linesToSplit,1)
                if iNewLine == 1
                    codeLines = [codeLines(1:iLine+iNewLine-1-1); linesToSplit{iNewLine}; codeLines(iLine+iNewLine:end)];
                else
                    codeLines = [codeLines(1:iLine+iNewLine-1-1); linesToSplit{iNewLine}; codeLines(iLine+iNewLine-1:end)];
                end
           end
       end
       linesToSplit = linesToSplit{1};
    end
    

   %  if contains(linesToSplit(2:end), ';') % QQ doesn't work yet: need to
   %  implement more complicated strategy to deal with multi-line matrices,
   %  vectors and text 
   % 
   %      % Indices of where to split the line
   %      splitIndices = [];
   %      splitLines = {};
   % 
   %      % Flags to keep track of context
   %      inDoubleQuotes = false;
   %      inSingleQuotes = false;
   %      inSquareBrackets = 0; % Using a counter for nested matrices
   %      inCurlyBraces = 0;    % Using a counter for nested cell arrays
   % 
   %      % Iterate through the line
   %      for i = 1:length(linesToSplit)
   %          switch linesToSplit(i)
   %              case '"'
   %                  inDoubleQuotes = ~inDoubleQuotes;
   %              case ''''
   %                  % MATLAB's syntax allows transpose using single quote. We need to ensure it's not a transpose
   %                  if i == length(linesToSplit) || linesToSplit(i+1) ~= ''''
   %                      inSingleQuotes = ~inSingleQuotes;
   %                  end
   %              case '['
   %                  inSquareBrackets = inSquareBrackets + 1;
   %              case ']'
   %                  inSquareBrackets = inSquareBrackets - 1;
   %              case '{'
   %                  inCurlyBraces = inCurlyBraces + 1;
   %              case '}'
   %                  inCurlyBraces = inCurlyBraces - 1;
   %              case ';'
   %                  if ~inDoubleQuotes && ~inSingleQuotes && inSquareBrackets == 0 && inCurlyBraces == 0
   %                      splitIndices = [splitIndices, i];
   %                  end
   %          end
   %      end
   %      % Split the line at the identified indices
   %      startIdx = 1;
   %      for idx = splitIndices
   %          splitLines{end+1} = linesToSplit(startIdx:idx);
   %          startIdx = idx + 1;
   %      end
   % 
   % 
   %      if size(splitLines,1) > 1
   %         splitLines = splitLines(arrayfun(@(x) ~isempty(splitLines{x}), 1:size(splitLines,1)));
   %         for iNewLine = 1:size(splitLines,1)
   %              if iNewLine == 1
   %                  codeLines = [codeLines(1:iLine+iNewLine-1-1); splitLines{iNewLine}; codeLines(iLine+iNewLine:end)];
   %              else
   %                  codeLines = [codeLines(1:iLine+iNewLine-1-1); splitLines{iNewLine}; codeLines(iLine+iNewLine-1:end)];
   %              end
   %         end
   %         linesToSplit = splitLines{1};
   %      end
   % 
   % end
   % 


    splitLine = linesToSplit;
end