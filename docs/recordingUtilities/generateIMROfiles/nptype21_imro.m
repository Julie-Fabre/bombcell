function nptype21_imro

% Build imro table for single shank NP 2.0 (probe type 21) for some useful
% patterns.
%
% Output is saved in the directory where this script is run.
%
% If a mulitple electrodes are mapped to the same channel, this program
% will not save the imro table.

% patternType = 0 for single bank "bankChoice"
% patternType = 1 for 192 rows starting from the parameter "botRow"
%                    which must be a multiple of 32 (0, 32, 64...448)
% patternType = 2 for checkboard covering two banks
% written by ???, modified by JF 

patternType = 2;
bankChoice = 1; % bank for patternType 0
botRow = 256;   % bottom row for patternType 1; must be a multiple of 32
botBank = 0;   % bottom bank for patternType 2; needs to be bank 0 or 1
refElec = 0;   % 0 for external, 1 for tip, 2 for site 127

bMapOK = 1;  

switch patternType
    case 0
        %single bank        
        nameStr = sprintf('NPtype21_bank%d_ref%d', bankChoice, refElec);
        %calculate electrode index for the 384 channels
        for i = 1:384
            bank(i) = bankChoice;
            chan(i) = i-1;
            elecInd(i) = chanToElec( bank(i), chan(i) );
            %fprintf("%d,%d,%d\n",bank(i),chan(i),elecInd(i));
        end
        
    case 1
        %384 channels starting from a specifed row, 0-448;
        %botRow must be a multiple of 32      
        nameStr = sprintf('NPtype21_botRow%d_ref%d', botRow, refElec);
        elecInd = botRow*2:(botRow*2 + 383);
        for i = 1:384
            [bank(i), chan(i)] = ElecToChan( elecInd(i) );
            if sum( chan(1:i-1)==chan(i) ) > 0
                fprintf( "duplicate channels => impossible map\n" );
                bMapOK = 0;
            end
            %fprintf("%d,%d,%d\n",bank(i),chan(i),elecInd(i));
        end
    case 2
        %checkerboard covering 2 full banks starting with botBank
        
        nameStr = sprintf('NPtype21_bank%d%d_checker_ref%d',botBank, botBank+1, refElec);
        % need to ensure that we pick non overlapping channels from
        % different 32 member channel groups.
        % Easy for two neighboring banks; patterns messy for 3 banks
        evenBankInd = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27];
        oddBankInd =  [2,3,4,5,10,11,12,13,18,19,20,21,26,27,28,29];
        bank(1:192) = botBank;
        bank(193:384) = botBank + 1;
        chanCount = 0;
        if mod(botBank,2) == 0
            % botBank is even
            for blockInd = 0:11     % first 192 chans, 16 per block
                elecInd(chanCount+1:chanCount+16) = botBank*384 + blockInd*32 + evenBankInd;
                chanCount = chanCount + 16;
            end
            for blockInd = 0:11
                elecInd(chanCount+1:chanCount+16) = (botBank+1)*384 + blockInd*32 + oddBankInd;
                chanCount = chanCount + 16;
            end
        else
            % botBank is odd
            for blockInd = 0:11     % first 192 chans, 16 per block
                elecInd(chanCount+1:chanCount+16) = botBank*384 + blockInd*32 + oddBankInd;
                chanCount = chanCount + 16;
            end
            for blockInd = 0:11
                elecInd(chanCount+1:chanCount+16) = (botBank+1)*384 + blockInd*32 + evenBankInd;
                chanCount = chanCount + 16;
            end
        end
        
        for i = 1:384
            [bank(i), chan(i)] = ElecToChan( elecInd(i) );
            if sum( chan(1:i-1)==chan(i) ) > 0
                fprintf( "duplicate channels => impossible map\n" );
                bMapOK = 0;
            end
            %fprintf("%d,%d,%d\n",bank(i),chan(i),elecInd(i));
        end        
    otherwise
        fprintf('unknown pattern type\n');
        return;
end

if bMapOK
    %open a new file wherever we are
    fileName = [nameStr,'.imro'];
    nmID = fopen(fileName,'w');

    % imro table
    % print first entry, specifying probe type and number of channels
    % for singly connected channels, the bank mask is 2^bank index. 
    % just make a lookup table:
    bankMask = [1, 2, 4, 8];
    [chan,isort] = sort(chan);
    bank = bank(isort);
    elecInd = elecInd(isort);
    fprintf(nmID,'(%d,%d)', 21, 384);
    for i = 1:numel(chan)
        fprintf(nmID,'(%d %d %d %d)', chan(i), bankMask(bank(i)+1), refElec, elecInd(i) );
    end
    fprintf(nmID, '\n');

    fclose(nmID);

    % plot the location of those electrodes using channels
    [~,~,~] = PlotElec21( bank, chan );
end

end

function [bank, chan] = ElecToChan( elecInd )

    % each bank has a unique factor and addend in the formula
    bF = [1,7,5,3];
    bA = [0,4,8,12]; 
    
    % left or right?
    bRight = mod(elecInd,2);  % Odd numbered electrodes = right
    
    bank = floor(elecInd/384); % elecInd = 0-1279, 3.3 banks of 384
    
    % block index = which block of 32 w/in the bank
    blockInd = floor((elecInd - bank*384)/32);
    subInd = elecInd - bank*384 - blockInd*32;
    %row range is 0-15 for left or for right
    row = floor(subInd/2);
    
    chan = 2 * mod((row*bF(bank+1) + bRight*bA(bank+1)),16) + bRight + 32*blockInd;

end

function [elecInd] = chanToElec( bank, chan )
    % each bank has a unique factor and addend in the formula
    bF = [1,7,5,3];
    bA = [0,4,8,12]; 
    
    % left or right?
    bRight = mod(chan,2);  % Odd numbered channels = right

    % channels occur in blocks of 32
    blockInd = floor(chan/32);
    
    % calculate the result of the modulo function (remainder)
    rem = (chan - 32*blockInd - bRight)/2;
    
    % To get the correct dividend, calculate the 15 possible dividends for
    % this bank and channel
    posibRow = 0:15;    %array of possible rows
    divArr = posibRow*bF(bank+1) + bRight*bA(bank+1);
    
    %calculate modulo 16 for all possible rows
    remArray = mod(divArr,16);
    
    %find the match
    row = posibRow(find(remArray==rem));
    
    elecInd = bank*384 + blockInd*32 + row*2 + bRight;
    
    
end

function [ chanPos, chanShank, refChan ] = PlotElec21( bank, chans )

    % NP 2.0 1 shank electode positions
    nElec = 1280;
    vSep = 15;   % in um
    hSep = 32;

    elecPos = zeros(nElec, 2);   

    elecPos(1:4:end,1) = hSep/2;           %sites 0,4,8...
    elecPos(2:4:end,1) =  (3/2)*hSep;      %sites 1,5,9...
    elecPos(3:4:end,1) = 0;                %sites 2,6,10...
    elecPos(4:4:end,1) =  hSep;            %sites 3,7,11...

    % fill in y values        
    viHalf = (0:(nElec/2-1))';                %row numbers
    elecPos(1:2:end,2) = viHalf * vSep;       %sites 0,2,4...
    elecPos(2:2:end,2) = elecPos(1:2:end,2);  %sites 1,3,5...

    % NP2.0 single shank reference electodes
    % there are 5 reference electrodes, but they can only be mapped to 192.
    refChan = [192];
    
    % One shank
    shank = ones(nElec,1);
    
    % get channel numbers and select AP only
    % chans = OriginalChans(meta);
    % [AP,LF,SY] = ChannelCountsIM(meta);
    % chans = (chans(1:AP))';
       
    % get bank indicies for all channels
    %[~,~, bank] = imroRead3A3B(meta);
    
    % chan to electrode for 
    for i = 1:numel(chans)
        elecInd(i) = chanToElec( bank(i), chans(i) );
    end
    
    chanPos = elecPos(elecInd+1,:);
    chanShank = shank(elecInd+1);
    
    % make a plot of all the electrode positions
    figure(1)
    scatter( elecPos(:,1), elecPos(:,2), 150, 'k', 'square' ); hold on;
    scatter( chanPos(:,1), chanPos(:,2), 100, 'b', 'square', 'filled' ); 
    xlim([-16,64]);
    ylim([-10,10000]);
    title('NP 2.0 single shank view');
    hold off;

end