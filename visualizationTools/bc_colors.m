function colorMtx =  bc_colors(nColors, backgroundCol)
%get distinguishable color-blind friendly colors 
% max 15 colors.

allColors =[255,255,255;...%1 white
            0,0,0;...%2 black
            37,37,37 ;...%3 dark gray
            103,103,103 ;...%4 light gray
            23,23,35;...%5 aubergine
            0,73,73;...%6 dark teal
            0,153,153;...%7 dark turquoise
            34,207,34;...%8 lime green
            73,0,146;...%9 indigo
            0,109,219;...%10 light royal blue
            182,109,255;...%11 amethyst
            255,109,182;...%12 Baker-Miller pink
            146,0,0;...%13 ruby
            143,78,0;...%14 saddle brown
            219,109,0;...%15 butternut orange
            255,223,77]./255;%16 daffodil yellow
if nColors ==2 
    colorMtx = allColors([10, 13],:,:);
elseif nColors < 13
    colorMtx = allColors(15-nColors:15,:,:); % exclude daffodil yellow, white and black 
elseif nColors == 14 
    colorMtx = allColors(3:16,:,:); % exclude daffodil yellow, white and black 
elseif nColors == 15
    if nargin > 1
        if backgroundCol == 'w'
            colorMtx = allColors(2:16,:,:); % exclude daffodil yellow, white and black 
        else
            colorMtx = allColors([1,3:16],:,:); % exclude daffodil yellow, white and black 
        end
    else %assume white background
        colorMtx = allColors(2:16,:,:); % exclude daffodil yellow, white and black 
        
    end

elseif nColors > 15 
    % qq make repeat
    if nargin > 1
        if backgroundCol == 'w'
            colorMtx = allColors(2:16,:,:); % exclude daffodil yellow, white and black 
        else
            colorMtx = allColors([1,3:16],:,:); % exclude daffodil yellow, white and black 
        end
    else %assume white background
        colorMtx = allColors(2:16,:,:); % exclude daffodil yellow, white and black 
        
    end
end









