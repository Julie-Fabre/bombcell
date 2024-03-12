function prettify_current_code()
% This function beautifies the content of the currently active MATLAB
% editor, based on rules outlined in formatRules.xml 
% ------
% Julie M. J. Fabre

    % get location of this script - the xml configuration will be in the
    % same spot
    currentPath = mfilename('fullpath');
    xmlPath = [currentPath, filesep, '..', filesep, 'formatRules.xml'];

    % Get the active editor
    activeEditor = matlab.desktop.editor.getActive();

    % Fetch the code from the active editor
    rawCode = activeEditor.Text;

    % Beautify the code (using our previous function)
    prettyCode = prettify_code(rawCode, xmlPath);

    % Update the content in the editor
    activeEditor.Text = prettyCode;

end
