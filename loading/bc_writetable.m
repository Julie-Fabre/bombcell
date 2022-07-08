function bc_writetable(T,filename,varargin)
%WRITETABLE Write a table to a file.
%
%   WRITETABLE(T) writes the table T to a comma-delimited text file. The file name is
%   the workspace name of the table T, appended with ".txt". If WRITETABLE cannot
%   construct the file name from the table input, it writes to the file "table.txt".
%
%   WRITETABLE overwrites any existing file by default, but this behavior can be
%   changed to append data or error using the "WriteMode" name-value pair.
%
%   WRITETABLE(T, FILENAME) writes the table T to the file FILENAME as column-oriented
%   data.
%
%   FILENAME can be one of these:
%
%       - For local files, FILENAME can contain an absolute file name with a
%         file extension. FILENAME can also be a relative path to the current
%         folder, or to a folder on the MATLAB path.
%         For example, to export to a file in the current folder:
%
%            writetable(T, "microscopy.txt");
%
%       - For remote files, FILENAME must be a full path specified as a
%         uniform resource locator (URL). For example, to export a remote
%         file to Amazon S3, specify the full URL for the file:
%
%            writetable(T, "s3://bucketname/path_to_file/microscopy.txt");
%
%         For more information on accessing remote data, see "Work with
%         Remote Data" in the documentation.
%
%   WRITETABLE(T, FILENAME, "FileType", FILETYPE) specifies the file type, where
%   FILETYPE is one of "text", "spreadsheet", "xml", or "auto".
%
%   The default value for FILETYPE is "auto", which makes WRITETABLE
%   detect the output file type from the file extension supplied in FILENAME.
%
%   WRITETABLE writes data to different file types as follows:
%
%   Delimited text files:
%   ---------------------
%
%   The following extensions are recognized: .txt, .dat, .csv, .log,
%                                            .text, .dlm
%
%   Writing to a delimited text file creates a column-oriented text
%   file, i.e., each column of each variable in T is written out as a
%   column in the file. The variable names of T are written out as
%   column headings in the first line of the file.
%
%   Use the following name-value pairs to control how data is written to
%   a delimited text file:
%
%   "Delimiter"          - The delimiter used in the file. Can be any of " ",
%                          "\t", ",", ";", "|". Default is ",".
%
%   "LineEnding"         - The characters to place as the end of a line.
%                          Defaults to "\r\n" on Windows and "\n" on Unix
%                          based platforms.
%
%   "WriteVariableNames" - A logical value that specifies whether or not
%                          the variable names of T are written out as
%                          column headings. Defaults to true.
%
%   "WriteRowNames"      - A logical value that specifies whether or not the row
%                          names of T are written out as the first column of the file.
%                          Default is false. If the "WriteVariableNames" and
%                          "WriteRowNames" parameter values are both true, the
%                          first dimension name of T is written out as the column
%                          heading for the first column of the file.
%
%   "QuoteStrings"       - A flag which specifies when to quote output text.
%                          - "minimal" (default) Any variables which contain
%                            the delimiter, line ending, or the double-quote
%                            character '"' will be quoted.
%                          - "all" All text, categorical, datetime, or
%                            duration variables will be quoted.
%                          - "none" No variables will be quoted.
%
%                          Quoted text will be surrounded by double-quote
%                          charcters (") and double-quote charcters already
%                          appearing in the data will be replaced with two
%                          double-quotes (i.e. ""). E.g. When quoted, 'a"b' will
%                          be written as '"a""b"'.
%
%   "DateLocale"         - The locale that WRITETABLE uses to create month and
%                          day names when writing datetimes to the file. LOCALE must
%                          be a character vector or scalar string in the form xx_YY.
%                          See the documentation for DATETIME for more information.
%
%   "Encoding"           - The encoding to use when creating the file.
%                          Default is "UTF-8".
%
%   "WriteMode"          - Append to an existing file or overwrite an
%                          existing file.
%                          - "overwrite" - Overwrite the file, if it exists.
%                                          This is the default option.
%                          - "append"    - Append to the bottom of the file,
%                                          if it exists.
%
%   Spreadsheet files:
%   ------------------
%
%   The following extensions are recognized: .xls, .xlsx, .xlsb, .xlsm,
%                                            .xltx, .xltm, .ods
%
%   WRITETABLE creates a column-oriented spreadsheet file, i.e., each column
%   of each variable in T is written out as a column in the file. The variable
%   names of T are written out as column headings in the first row of the file.
%
%   Use the following name-value pairs to control how data is written
%   to a spreadsheet file:
%
%   "WriteVariableNames" - A logical value that specifies whether or not the
%                          variable names of T are written out as column headings.
%                          Defaults to true.
%
%   "WriteRowNames"      - A logical value that specifies whether or not the row
%                          names of T are written out as first column of the specified
%                          region of the file. Defaults to false. If the
%                          "WriteVariableNames" and "WriteRowNames" parameter values
%                          are both true, the first dimension name of T is written out as
%                          the column heading for the first column.
%
%   "DateLocale"         - The locale that WRITETABLE uses to create month and day
%                          names when writing datetimes to the file. LOCALE must be
%                          a character vector or scalar string in the form xx_YY.
%                          Note: The "DateLocale" parameter value is ignored
%                          whenever dates can be written as Excel-formatted dates.
%
%   "Sheet"              - The sheet to write, specified as either the worksheet name,
%                          or a positive integer indicating the worksheet index.
%
%   "Range"              - A character vector or scalar string that specifies a
%                          rectangular portion of the worksheet to write, using the
%                          Excel A1 reference style.
%
%   "UseExcel"           - A logical value that specifies whether or not to create the
%                          spreadsheet file using Microsoft(R) Excel(R) for Windows(R).
%                          Set "UseExcel" to one of these values:
%                          - false: Does not open an instance of Microsoft Excel
%                                   to write the file. This is the default setting.
%                                   This setting may cause the data to be
%                                   written differently for files with live updates
%                                   (e.g. formula evaluation or plugins).
%                          - true:  Opens an instance of Microsoft Excel to write
%                                   the file on a Windows system with Excel installed.
%
%   "WriteMode"          - Perform an in-place write, append to an existing
%                          file or sheet, overwrite an existing file or
%                          sheet.
%                          - "inplace":        In-place replacement of the
%                                              data in the sheet.
%                                              This is the default option.
%                          - "overwritesheet": If sheet exists, overwrite
%                                              contents of sheet.
%                          - "replacefile":    Create a new file. Prior
%                                              contents of the file and all
%                                              the sheets are removed.
%                          - "append":         Append to the bottom of the
%                                              occupied range within the
%                                              sheet.
%
%   "AutoFitWidth"       - A logical value that specifies whether or not to change
%                          column width to automatically fit the contents. Defaults to true.
%
%   "PreserveFormat"     - A logical value that specifies whether or not to preserve
%                          existing cell formatting. Defaults to true.
%
%   XML files:
%   ----------
%
%   The following extensions are recognized: .xml
%
%   Tabular structure present within an XML file:
%
%       <table> ----------------------------- Table Node
%           <row> --------------------------- Row Node
%               <date>2019-07-11</date> ----- Variable Node
%               <index>8191</index>
%               <name>Lorem</name>
%           </row>
%           <row>
%               <date>2020-01-04</date>
%               <index>131071</index>
%               <name>Ipsum</name>
%           </row>
%       </table>
%
%   Writing to an XML file creates a node in the file for each row of T.
%   Variable names are used to label the child nodes under the row nodes.
%
%   Use the following name-value pairs to control how data is written to
%   an XML file:
%
%   "RowNodeName"     - Node name which delineates rows of the table in
%                       the output XML file. Defaults to "row".
%
%   "TableNodeName"   - Name to use for the document root node of the
%                       output XML file. Defaults to "table".
%
%   "AttributeSuffix" - Variable name suffix indicating variables to
%                       write out as XML attributes. WRITETABLE will
%                       write out all variables with the specified
%                       suffix as XML attributes, excluding the suffix
%                       in the resulting attribute name. Defaults
%                       to "Attribute".
%
%   "WriteRowNames"   - Logical value indicating whether to write out the
%                       row names of T to the XML file. Defaults to false.
%
%   "DateLocale"      - The locale that WRITETABLE uses to create month and
%                       day names when writing datetimes to the file. LOCALE must
%                       be a character vector or scalar string in the form xx_YY.
%                       See the documentation for DATETIME for more information.
%
%   In some cases, WRITETABLE creates a file that does not represent T exactly, as
%   described below. If you use READTABLE(FILENAME) to read that file back in and create
%   a new table, the result may not have exactly the same format or contents as the
%   original table.
%
%   *  WRITETABLE writes out numeric variables using long g format, and
%      categorical or character variables as unquoted text.
%   *  For non-character variables that have more than one column, WRITETABLE
%      writes out multiple delimiter-separated fields on each line, and constructs
%      suitable column headings for the first line of the file.
%   *  WRITETABLE writes out variables that have more than two dimensions as two
%      dimensional variables, with trailing dimensions collapsed.
%   *  For cell-valued variables, WRITETABLE writes out the contents of each cell
%      as a single row, in multiple delimiter-separated fields, when the contents are
%      numeric, logical, character, or categorical, and writes out a single empty
%      field otherwise.
%
%   Save T as a mat file if you need to import it again as a table.
%
%   See also READTABLE, WRITETIMETABLE, TABLE

% Copyright 2012-2021 The MathWorks, Inc.

    import matlab.io.internal.utility.suggestWriteFunctionCorrection
    import matlab.io.internal.validators.validateWriteFunctionArgumentOrder
    import matlab.io.internal.vfs.validators.validateCloudEnvVariables;
    import matlab.internal.datatypes.validateLogical

    if nargin == 0
        error(message("MATLAB:minrhs"));
    elseif nargin == 1
        tablename = inputname(1);
        if isempty(tablename)
            tablename = "table";
        end
        filename = tablename + ".txt";
    end

    validateWriteFunctionArgumentOrder(T, filename, "writetable", "table", @istable);
    [T, filename, varargin{:}] = convertStringsToChars(T,filename,varargin{:});

    if ~istable(T)
        suggestWriteFunctionCorrection(T, "writetable");
    end

    if isempty(filename) || ~ischar(filename)
        error(message("MATLAB:virtualfileio:path:cellWithEmptyStr","FILENAME"));
    end

    % second input is not really optional with NV-pairs.
    if nargin > 2 && mod(nargin,2) > 0
        error(message("MATLAB:table:write:NoFileNameWithParams"));
    end
    try
        if nargin < 2 || isempty(filename)
            type = "text";
            tablename = inputname(1);
            if isempty(tablename)
                tablename = class(T);
            end
            filename = tablename +".txt";
            suppliedArgs = {"WriteVariableNames",true,"WriteRowNames",false};
        else
            pnames = {'FileType'};
            dflts =  {   [] };
            [type,supplied,suppliedArgs] = matlab.internal.datatypes.parseArgs(pnames, dflts, varargin{:});
            [~,name,ext] = fileparts(filename);

            if isempty(name)
                error(message("MATLAB:table:write:NoFilename"));
            end

            if ~supplied.FileType || (ischar(type) && strcmp(type, "auto"))
                if isempty(ext)
                    ext = ".txt";
                    filename = filename + ext;
                end
                switch lower(ext)
                  case {'.txt' '.dat' '.csv', '.tsv', '.htsv'}, type = 'text';
                  case {'.xls' '.xlsx' '.xlsb' '.xlsm' '.xltx' '.xltm'}, type = 'spreadsheet';
                  case {'.xml'}, type = 'xml';
                  otherwise
                    error(message("MATLAB:table:write:UnrecognizedFileExtension",ext));
                end
            elseif ~ischar(type) && ~(isstring(type) && isscalar(type))
                error(message("MATLAB:textio:textio:InvalidStringProperty","FileType"));
            else
                fileTypes = {'text' 'spreadsheet' 'xml'};
                itype = find(strncmpi(type,fileTypes,strlength(type)));
                if isempty(itype)
                    error(message("MATLAB:table:write:UnrecognizedFileType",type));
                elseif ~isscalar(itype)
                    error(message("MATLAB:table:write:AmbiguousFileType",type));
                end
                type = fileTypes{itype};
                % Add default extension if necessary
                if isempty(ext)
                    dfltFileExts = {'.txt' '.xls', '.xml'};
                    ext = dfltFileExts{itype};
                    filename = [filename ext];
                end
            end
        end

        if type == "text" || type == "spreadsheet"
            % Check WriteMode to determine whether the file should be
            % appended or overwritten
            pnames = {'WriteMode','WriteVariableNames','WriteRowNames'};
            if type == "text"
                dflts =  {'overwrite',true,false};
                validModes = ["overwrite","append"];
            else
                dflts = {'inplace',true,false};  % validation of WriteMode values
                validModes = ["inplace","overwritesheet","append","replacefile"];
            end
            [sharedArgs.WriteMode, sharedArgs.WriteVarNames, sharedArgs.WriteRowNames, ...
             supplied, remainingArgs]...
                = matlab.internal.datatypes.parseArgs(pnames, dflts, suppliedArgs{:});

            sharedArgs.WriteMode = validatestring(sharedArgs.WriteMode,validModes);
            sharedArgs.WriteRowNames = validateLogical(sharedArgs.WriteRowNames, ...
                                                       "WriteRowNames") && ~isempty(T.Properties.RowNames);
            sharedArgs.WriteVarNames = validateLogical(sharedArgs.WriteVarNames, ...
                                                       "WriteVariableNames");

            % Setup LocalToRemote object with a remote folder.
            % check if the file exists remotely, we need to download since we
            % will append to file
            if type == "spreadsheet" && any(sharedArgs.WriteMode == ...
                                            ["inplace", "append", "overwritesheet"])
                try
                    remote2Local = matlab.io.internal.vfs.stream.RemoteToLocal(filename);
                    tempFile = remote2Local.LocalFileName;
                catch ME
                    if contains(ME.identifier,"EnvVariablesNotSet")
                        throwAsCaller(ME);
                    end
                    remote2Local = [];
                    tempFile = filename;
                end
            end

            validURL = matlab.io.internal.vfs.validators.isIRI(char(filename));
            if validURL
                % check whether credentials are set
                validateCloudEnvVariables(filename);

                % remote write for spreadsheets
                if type == "spreadsheet"
                    filenameWithoutPath = matlab.io.internal.vfs.validators.IRIFilename(filename);
                    remoteFolder = extractBefore(filename,filenameWithoutPath);
                    ext = strfind(filenameWithoutPath,".");
                    if ~isempty(ext)
                        index = ext(end)-1;
                        ext = extractAfter(filenameWithoutPath,index);
                        filenameWithoutPath = extractBefore(filenameWithoutPath,index+1);
                    end
                    ltr = matlab.io.internal.vfs.stream.LocalToRemote(remoteFolder);
                    if ~isempty(remote2Local) && any(type == ["spreadsheet","text"])
                        % file exists in remote location, append to file
                        ltr.CurrentLocalFilePath = tempFile;
                        ltr.setRemoteFileName(filenameWithoutPath, ext);
                    else
                        % file does not exist in remote location
                        setfilename(ltr,filenameWithoutPath,ext);
                    end
                    fname = ltr.CurrentLocalFilePath;
                    validateRemotePath(ltr);
                else
                    % text files now have native cloud access support
                    fname = filename;
                end
            elseif matlab.io.internal.vfs.validators.hasIriPrefix(filename) && ~validURL
                % possibly something wrong with the URL such as a malformed Azure
                % URL missing the container@account part
                error(message("MATLAB:virtualfileio:stream:invalidFilename", filename));
            else
                [path, ~, ext] = fileparts(filename);
                if isempty(path)
                    % If no path is passed in, we should assume the current
                    % directory is the one we are using. Later on, some more general
                    % code will look for the existing file, but it does path lookup
                    % to match the file name.
                    fname = fullfile(pwd,filename);
                else
                    % In the case of a full/partial path, no path lookup will
                    % happen.
                    fname = filename;
                end
            end

            sharedArgs.SuppliedWriteVarNames = supplied.WriteVariableNames;

            switch lower(type)
                case "text"
                    matlab.io.internal.writing.writeTextFile(T,fname,sharedArgs,remainingArgs);
                case "spreadsheet"
                    matlab.io.internal.writing.writeXLSFile(T,fname,ext(2:end),sharedArgs,remainingArgs);
                otherwise
                    error(message('MATLAB:table:write:UnrecognizedFileType',type));
            end

            if validURL && type == "spreadsheet"
                % upload the file to the remote location
                upload(ltr);
            end
        elseif type == "xml"
            % currently no shared args with text or spreadsheet files, VFS
            % support is implemented in c++
            matlab.io.xml.internal.write.writeTable(T,filename,suppliedArgs);
        end
    catch ME
        if strcmp(ME.identifier, "MATLAB:fileparts:MustBeChar")
            throwAsCaller(MException(message("MATLAB:virtualfileio:path:cellWithEmptyStr","FILENAME")));
        elseif strcmp(ME.identifier, "MATLAB:FileIO:InvalidRemoteLocation")
            ME = MException("MATLAB:virtualfileio:stream:invalidFilename", ...
                            "Input location does not exist.");
            error(matlab.io.internal.vfs.util.convertStreamException(ME, fname));
        elseif strcmp(ME.identifier, "MATLAB:FileIO:HadoopNotInitialized")
            ME = MException("MATLAB:virtualfileio:hadooploader:hadoopNotFound", ...
                            "Hadoop credentials not found.");
            error(matlab.io.internal.vfs.util.convertStreamException(ME, fname));
        elseif strcmp(ME.identifier, "MATLAB:FileIO:InvalidURLScheme")
            ME = MException("MATLAB:virtualfileio:stream:CannotFindLocation", ...
                            "Invalid scheme.");
            error(matlab.io.internal.vfs.util.convertStreamException(ME, fname));
        elseif strcmp(ME.identifier, "MATLAB:badfid_mx")
            error(message("MATLAB:fopen:InvalidFileLocation"));
        elseif exist("ltr","var")
            error(matlab.io.internal.vfs.util.convertStreamException(ME, remoteFolder));
        else
            throwAsCaller(ME);
        end
    end
