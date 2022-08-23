allenCCF_path = fileparts(which('allenCCFbregma'));
cmap_filename = [allenCCF_path filesep 'allen_ccf_colormap_2017.mat'];
load(cmap_filename);
slicec = 150; 

I = reg(:,1:floor(size(reg,2)/2),547-slicec);
mask = tv_crop(:,floor(size(tv_crop,2)/2):size(tv_crop,2),slicec);
figure();
ax1 = axes;

A = imagesc([1, floor(size(reg,2)/2)],[],I);
hold on; 

hold on;
C = imagesc([1, floor(size(reg,2)/2)],[],av_crop(:,1:floor(size(reg,2)/2),slicec));
C.AlphaData = 0.8;
xlim([0 floor(size(reg,2)/2)+size(mask,2)])
ylim([ 0 size(mask,1)])
axis equal;

ax2 = axes;
B = imagesc([floor(size(reg,2)/2), floor(size(reg,2)/2)+size(mask,2)],[],mask); 
linkaxes([ax1,ax2]);
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
colormap(ax1,cmap);
colormap(ax2,'hot');
%colormap(ax3,cmap);
set(ax2,'color','none','visible','off');
xlim([0 floor(size(reg,2)/2)+size(mask,2)])
ylim([ 0 size(mask,1)])
axis equal;
