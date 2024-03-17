function trimedLine = prettify_trim_white_space(lineToTrim)
    lineToTrim = regexprep(lineToTrim, '^[ \t]+', ''); % leading white space
    trimedLine = regexprep(lineToTrim, '^[ \t]+$', ''); % trailing white space
end