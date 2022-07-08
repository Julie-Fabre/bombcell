function t = bc_readtable(filename,varargin)
%READTABLE Create a table by reading from a file.
%
%   Use the READTABLE function to create a table by reading column-oriented
%   data from a file. READTABLE automatically determines the file format
%   from its extension as described below.
%
%   T = READTABLE(FILENAME) creates a table by reading from a file, where
%   FILENAME can be one of these:
%
%       - For local files, FILENAME can be a full path that contains
%         a filename and file extension. FILENAME can also be a relative
%         path to the current folder, or to a folder on the MATLAB path.
%         For example, to import a file on the MATLAB path:
%
%            T = readtable("patients.xls");
%
%       - For files from an Internet URL or stored at a remote location,
%         FILENAME must be a full path using a Uniform Resource Locator
%         (URL). For example, to import a remote file from Amazon S3,
%         specify the full URL for the file:
%
%            T = readtable("s3://bucketname/path_to_file/my_table.xls");
%
%         To read tabular data from a web page, specify the URL; if the
%         URL points to an HTML resource, but does not end in ".html" 
%         or ".htm", also specify the file type:
%
%            url = "https://www.mathworks.com/matlabcentral/cody/groups/78";
%            T = readtable(url,"FileType","html");
%
%         For more information on accessing remote data, see "Work with
%         Remote Data" in the documentation.
%
%   T = READTABLE(FILENAME,"FileType",FILETYPE) specifies the file type, where
%   FILETYPE is one of "text", "delimitedtext", "fixedwidth", "spreadsheet",
%   "xml", "html", or "worddocument".
%
%   T = READTABLE(FILENAME,OPTS) creates a table by reading from a file stored
%   at FILENAME using the supplied ImportOptions OPTS. OPTS specifies variable
%   names, selected variable names, variable types, and other information regarding
%   the location of the data.
%
%   For example, import a subset of the data in a file:
%
%       opts = detectImportOptions("patients.xls");
%       opts.SelectedVariableNames = ["Systolic","Diastolic"];
%       T = readtable("patients.xls",opts)
%
%   READTABLE reads data from different file types as follows:
%
%   Text files (delimited and fixed-width): 
%
%     The following extensions are supported: .txt, .dat, .csv, .log,
%                                             .text, .dlm
%
%     Reading from a delimited text file creates one variable in T for each
%     column in the file. Variable names can be taken from the first row of
%     the file. By default, the variables created are either double, if the
%     column is primarily numeric, or datetime, duration, or text etc. If
%     data in a column cannot be converted to numeric, datetime or
%     duration, the column is imported as text.
%
%   Spreadsheet files:
%
%     The following extensions are supported: .xls, .xlsx, .xlsb, .xlsm, 
%                                             .xltm, .xltx, .ods
%
%     Reading from a spreadsheet file creates one variable in T for each
%     column in the file. By default, the variables created are either
%     double, datetime or text--depending on the type in the file.
%
%     READTABLE converts both empty fields or cells and values which cannot
%     be converted to the expected type to:
%       - NaN (for a numeric or duration variable),
%       - NaT (for a datetime variable),
%       - Empty character vector ('') or missing string (for text variables).
%
%   Word documents:
%
%     The following extensions are supported: .docx
%
%     Reading from a Word document file imports data from a table. Each column
%     in the table creates one variable in T. Variable names can be taken from
%     the first row of the table. By default, the variables created are either
%     double, if the column is primarily numeric, or datetime, duration, or
%     text etc. If data in a column cannot be converted to numeric, datetime
%     or duration, the column is imported as text. The default data type for
%     text import is string.
%
%   HTML files: 
%
%     The following extensions are supported: .html, .xhtml, .htm
%
%     Reading from an HTML file imports data from a <TABLE> element. Each
%     column in the table creates one variable in T. Variable names can be
%     taken from the first row of the table. By default, the variables created
%     are either double, if the column is primarily numeric, or datetime,
%     duration, or text etc. If data in a column cannot be converted to
%     numeric, datetime or duration, the column is imported as text. The
%     default data type for text import is string.
%
%   XML files:
%
%     The following extensions are supported: .xml
%
%     Tabular structure present within an XML file:
%
%         <table> ----------------------------- Table Node
%             <row> --------------------------- Row Node
%                 <date>2019-07-11</date> ----- Variable Node
%                 <index>8191</index>
%                 <name>Lorem</name>
%             </row>
%             <row>
%                 <date>2020-01-04</date>
%                 <index>131071</index>
%                 <name>Ipsum</name>
%             </row>
%         </table>
%
%     Reading from an XML file creates one row in T for each repeated node
%     in the file that is detected under the table node. Variable names are
%     taken from the names of the child nodes under the row nodes in the file.
%
%   Name-Value Pairs for ALL file types:
%   ------------------------------------
%
%   "FileType"              - Specify the file as "text", "delimitedtext",
%                             "fixedwidth", "spreadsheet", "xml", "html",
%                             or "worddocument".
%
%   "VariableNamingRule"    - A character vector or a string scalar that
%                             specifies how the output variables are named.
%                             It can have either of the following values:
%
%                             "modify"   Modify variable names to make them
%                                        valid MATLAB Identifiers.
%                                        (default)
%                             "preserve" Preserve original variable names
%                                        allowing names with spaces and
%                                        non-ASCII characters.
%
%   "MissingRule"           - Rules for interpreting missing or
%                             unavailable data:
%                             "fill"      Replace missing data with the
%                                         contents of the "FillValue"
%                                         property.
%                             "error"     Stop importing and display an
%                                         error message showing the missing
%                                         record and field.
%                             "omitrow"   Omit rows that contain missing
%                                         data.
%                             "omitvar"   Omit variables that contain
%                                         missing data.
%
%   "ImportErrorRule"       - Rules for interpreting nonconvertible
%                             or bad data:
%                             "fill"      Replace the data where errors
%                                         occur with the contents of the
%                                         "FillValue" property.
%                             "error"     Stop importing and display an
%                                         error message showing the
%                                         error-causing record and field.
%                             "omitrow"   Omit rows where errors occur.
%                             "omitvar"   Omit variables where errors
%                                         occur.
%
%   "ReadRowNames"          - Whether or not to import the first variable
%                             as row names. Defaults to false.
%
%   "TreatAsMissing"        - Text which is used in a file to represent
%                             missing data, e.g. "NA".
%
%   "TextType"              - The type to use for text variables, specified
%                             as "char" or "string".
%
%   "DatetimeType"          - The type to use for date variables, specified
%                             as "datetime", "text", or "exceldatenum".
%                             Defaults to "datetime".
%
%   "WebOptions"            - HTTP(s) request options, specified as a 
%                             weboptions object. 
%
%   Name-Value Pairs for TEXT and SPREADSHEET only:
%   -----------------------------------------------
%
%   "Range"                 - The range to consider when detecting data.
%                             Specified using any of the following syntaxes:
%                             - Starting cell: A string or character vector
%                               containing a column letter and a row number,
%                               or a 2 element numeric vector indicating
%                               the starting row and column.
%                             - Rectangular range: A start and end cell separated
%                               by colon, e.g. "C2:N15", or a four element
%                               numeric vector containing start row, start
%                               column, end row, end column, e.g. [2 3 15 13].
%                             - Row range: A string or character vector
%                               containing a starting row number and ending
%                               row number, separated by a colon.
%                             - Column range: A string or character vector
%                               containing a starting column letter and
%                               ending column letter, separated by a colon.
%                             - Starting row number: A numeric scalar
%                               indicating the first row where data is found.
%
%   "NumHeaderLines"        - The number of header lines in the file.
%
%   "ExpectedNumVariables"  - The expected number of variables.
%
%   "ReadVariableNames"     - Whether or not to expect variable names in
%                             the file. Defaults to true.
%
%   Name-Value Pairs for TEXT, XML, HTML, and Word documents only:
%   --------------------------------------------------------------
%
%   "DateLocale"         - The locale used to interpret month and day
%                          names in datetime text. Must be a character
%                          vector or scalar string in the form xx_YY.
%                          See the documentation for DATETIME for more
%                          information.
%
%   "DecimalSeparator"   - Character used to separate the integer part
%                          of a number from the decimal part of the
%                          number.
%
%   "ThousandsSeparator" - Character used to separate the thousands
%                          place digits.
%
%   Name-Value Pairs for TEXT, XML, and HTML only:
%   ----------------------------------------------
%
%   "Encoding"           - The character encoding scheme associated with
%                          the file.
%
%   Name-Value Pairs for TEXT and XML only:
%   ---------------------------------------
%
%   "DurationType"       - The type to use for duration, specified as
%                          "duration" or "text". Defaults to "duration".
%
%   "Whitespace"         - Characters to treat as whitespace.
%
%   "TrimNonNumeric"     - Whether or not to remove nonnumeric characters
%                          from a numeric variable. Defaults to false.
%
%   "HexType"            - Set the output type of a hexadecimal
%                          variable.
%
%   "BinaryType"         - Set the output type of a binary variable.
%
%   "CollectOutput"      - Whether or not to concatenate consecutive output
%                          of the same MATLAB class into a single array.
%                          Defaults to false.
%
%   Name-Value Pairs for TEXT, HTML, and Word documents only:
%   ---------------------------------------------------------
%
%   "RowNamesColumn"     - The column where the row names are
%                          located.
%
%   Name-Value Pairs for TEXT only:
%   -------------------------------
%
%   "Delimiter"                 - Field delimiter characters in a delimited
%                                 text file, specified as a character
%                                 vector, string scalar, cell array of
%                                 character vectors, or string array.
%
%   "CommentStyle"              - Style of comments, specified as a
%                                 character vector, string scalar, cell
%                                 array of character vectors, or string
%                                 array.
%
%   "LineEnding"                - End-of-line characters, specified as a
%                                 character vector, string scalar, cell
%                                 array of character vectors, or string
%                                 array.
%
%   "ConsecutiveDelimitersRule" - Rule to apply to fields containing
%                                 multiple consecutive delimiters:
%                                 "split"     Split consecutive delimiters
%                                             into multiple fields.
%                                 "join"      Join the delimiters into one
%                                             single delimiter.
%                                 "error"     Ignore consecutive delimiters
%                                             during detection (treated as
%                                             "split"), but the
%                                             resulting read will error.
%
%   "LeadingDelimitersRule"     - Rule to apply to delimiters at the
%                                 beginning of a line:
%                                 "keep"      Keep leading delimiters.
%                                 "ignore"    Ignore leading delimiters.
%                                 "error"     Ignore leading delimiters
%                                             during detection, but the
%                                             resulting read will error.
%
%   "TrailingDelimiterRule"     - Rule to apply to delimiters at the
%                                 end of a line:
%                                 "keep"      Keep trailing delimiters.
%                                 "ignore"    Ignore trailing delimiters.
%                                 "error"     Ignore trailing delimiters
%                                             during detection, but the
%                                             resulting read will error.
%
%   "VariableWidths"            - Widths of the variables for a fixed width
%                                 file.
%
%   "EmptyLineRule"             - Rule to apply to empty lines in the file:
%                                 "skip"      Skip empty lines.
%                                 "read"      Read empty lines.
%                                 "error"     Ignore empty lines during
%                                             detection, but the resulting
%                                             read will error.
%
%   "VariableNamesLine"         - The line where the variable names are
%                                 located.
%
%   "PartialFieldRule"          - Rule to handle partial fields in the data:
%                                 "keep"      Keep the partial field data
%                                             and convert the text to the
%                                             appropriate data type.
%                                 "fill"      Replace missing data with the
%                                             contents of the "FillValue"
%                                             property.
%                                 "omitrow"   Omit rows that contain
%                                             partial data.
%                                 "omitvar"   Omit variables that contain
%                                             partial data.
%                                 "wrap"      Begin reading the next line
%                                             of characters.
%                                 "error"     Ignore partial field data
%                                             during detection, but the
%                                             resulting read will error.
%
%   "VariableUnitsLine"         - The line where the variable units are
%                                 located.
%
%   "VariableDescriptionsLine"  - The line where the variable descriptions
%                                 are located.
%
%   "ExtraColumnsRule"          - Rule to apply to extra columns of data
%                                 that appear after the expected variables:
%                                 "addvars"   Creates new variables to 
%                                             import extra columns. If there
%                                             are N extra columns, then import
%                                             new variables as "ExtraVar1",
%                                             "ExtraVar2",..., "ExtraVarN".
%                                 "ignore"    Ignore the extra columns of
%                                             data.
%                                 "wrap"      Wrap the extra columns of
%                                             data to new records.
%                                 "error"     Display an error message and
%                                             abort the import operation.
%
%   Name-Value Pairs for SPREADSHEET only:
%   --------------------------------------
%
%   "UseExcel"                  - Whether or not to read the spreadsheet
%                                 file using Microsoft(R) Excel(R) on 
%                                 Windows(R):
%                                 true  - Opens an instance of Microsoft Excel
%                                         to read the file on a Windows system
%                                         with Excel installed.
%                                 false - Does not open an instance of Microsoft
%                                         Excel to read the file. This is the
%                                         default setting.
%
%   "Sheet"                     - The sheet from which to read the table.
%
%   "DataRange"                 - Where the table data is located.
%
%   "RowNamesRange"             - Where the row names are located.
%
%   "VariableNamesRange"        - Where the variable names are located.
%
%   "VariableUnitsRange"        - Where the variable units are located.
%
%   "VariableDescriptionsRange" - Where the variable descriptions are
%                                 located.
%
%   Name-Value Pairs for HTML and Word documents only:
%   --------------------------------------------------
%
%   "TableIndex"                - Integer selection which table to extract.
%
%   "TableSelector"             - XPath expression that selects the table
%                                 to extract.
%
%   "VariableNamesRow"          - The row where the variable names are
%                                 located.
%
%   "VariableUnitsRow"          - The row where the variable units are
%                                 located.
%
%   "VariableDescriptionsRow"   - The row where the variable descriptions
%                                 are located.
%
%   "EmptyRowRule"              - Rule to apply to empty lines in the file:
%                                 "skip"      Skip empty lines.
%                                 "read"      Read empty lines.
%                                 "error"     Ignore empty lines during
%                                             detection, but the resulting
%                                             read will error.
%
%   "EmptyColumnRule"           - Rule to apply to empty columns in the file:
%                                 "skip"      Skip empty columns.
%                                 "read"      Read empty columns.
%                                 "error"     Error on empty columns.
%
%   Name-Value Pairs for XML only:
%   ------------------------------
%
%   "RowNodeName"                  - Node name which delineates rows of
%                                    the output table.
%
%   "RowSelector"                  - XPath expression that selects the XML
%                                    Element nodes which delineate rows of
%                                    the output table.
%
%   "VariableNodeNames"            - Node names which will be treated as
%                                    variables of the output table.
%
%   "VariableSelectors"            - XPath expressions that select the XML
%                                    Element nodes to be treated as variables
%                                    of the output table.
%
%   "TableNodeName"                - Name of the node which contains table
%                                    data. If multiple nodes have the same
%                                    name, READTABLE uses the first node
%                                    with that name.
%
%   "TableSelector"                - XPath expression that selects the XML
%                                    Element node containing the table data.
%
%   "VariableUnitsSelector"        - XPath expression that selects the XML
%                                    Element nodes containing the variable
%                                    units.
%
%   "VariableDescriptionsSelector" - XPath expression that selects the XML
%                                    Element nodes containing the variable
%                                    descriptions.
%
%   "RowNamesSelector"             - XPath expression that selects the XML
%                                    Element nodes containing the row names.
%
%   "RepeatedNodeRule"             - Rule for managing repeated nodes in a
%                                    given row of a table:
%                                    "addcol"     Add a column for each
%                                                 repeated node.
%                                    "ignore"     Ignore repeated nodes.
%                                    "error"      Ignore repeated nodes
%                                                 during detection, but the
%                                                 resulting read will error.
%
%   "ImportAttributes"             - Import XML node attributes as variables
%                                    of the output table. Defaults to true.
%
%   "AttributeSuffix"              - Suffix to append to all output table
%                                    variable names corresponding to
%                                    attributes in the XML file. Defaults
%                                    to "Attribute".
%
%   "RegisteredNamespaces"         - The namespace prefixes that are mapped
%                                    to namespace URLs for use in selector
%                                    expressions.
%
%   Name-Value Pairs supported with Text and Spreadsheet Import Options OPTS:
%   -------------------------------------------------------------------------
%
%       Supported for all file types:
%         "WebOptions" -   HTTP(s) request options, specified as a 
%                          weboptions object.
%
%   These have slightly different behavior when used with import options:
%
%       T = readtable(FILENAME, OPTS, "Name1", Value1, "Name2", Value2, ...)
%
%         "ReadVariableNames" true  - Reads the variable names from the
%                                     opts.VariableNamesRange or opts.VariableNamesLine
%                                     location.
%                             false - Uses variable names from the import options.
%
%         "ReadRowNames"      true  - Reads the row names from the opts.RowNamesRange
%                                     or opts.RowNamesColumn location.
%                             false - Does not import row names.
%
%       Text only parameters:
%         "DateLocale" - Override the locale used when importing dates.
%         "Encoding"   - Override the encoding defined in import options.
%
%       Spreadsheet only parameters:
%         "Sheet"      - Override the sheet value in the import options.
%         "UseExcel"   - Same behavior as READCELL without import options.
%
%   See also WRITETABLE, READTIMETABLE, READMATRIX, READCELL, TABLE,
%            DETECTIMPORTOPTIONS.

%   Copyright 2012-2021 The MathWorks, Inc.

if any(cellfun(@(arg) isa(arg,"matlab.io.ImportOptions"),varargin),'all')
    error(message('MATLAB:textio:io:OptsSecondArg','readtable'))
end

[varargin{1:2:end}] = convertStringsToChars(varargin{1:2:end});
names = varargin(1:2:end);

try
        t = bc_legacyReadtable(filename,varargin);
   
catch ME
    throw(ME)
end
