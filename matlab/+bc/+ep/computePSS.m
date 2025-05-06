 function pss = computePSS(thisACG)
 pss = find(thisACG(500:1000) >= ...
        nanmean(thisACG( 600:900))); % nanmean(ephysParams.ACG(iUnit, 900:1000)) also works. 
    if ~isempty(pss)
        pss = pss(1);
    else
        pss = NaN;
    end
end
    