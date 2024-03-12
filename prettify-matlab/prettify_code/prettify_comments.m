function formatedComment = prettify_comments(rawComment)

        if startsWith(rawComment, '%%') % Handling bold comments
            commentPart = ['%%', strtrim(rawComment(2:end))];
            if length(commentPart) > 2 % If there's text after the '%%'
                formatedComment = [commentPart(1:2), ' ', commentPart(3:end)]; % Ensuring a single space after the '%%'
            end
        else
            commentPart = ['%', strtrim(rawComment(2:end))]; % Regular comments
            if length(commentPart) > 1 % If there's text after the '%'
                formatedComment = [commentPart(1), ' ', commentPart(2:end)]; % Ensuring a single space after the '%'
            end
        end
end