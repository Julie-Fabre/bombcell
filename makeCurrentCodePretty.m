function makeCurrentCodePretty
 % This function beautifies the content of the currently active MATLAB editor.
 
 import matlab.desktop.editor. * ;
 
 % get location of this script - the xml configuration will be in the
 % same spot
 currentPath = mfilename('fullpath');
 xmlPath = [currentPath, filesep, '..', filesep, 'formatRules.xml'];
 
 % Get the active editor
 activeEditor = matlab.desktop.editor.getActive();
 
 % Fetch the code from the active editor
 rawCode = activeEditor.Text;
 
 % Beautify the code (using our previous function)
 prettyCode = makeCodePretty(rawCode, xmlPath);
 
 % Update the content in the editor
 activeEditor.Text = prettyCode;
end
