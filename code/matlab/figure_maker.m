function figure_maker(fig_cell, fig_title)

global fn;

color=[1.0 1.0 1.0];

fontname = 'Times'; %'Times', 'Helvetica'
set(0,'defaultaxesfontname',fontname);
set(0,'defaulttextfontname',fontname);
set(0,'defaulttextfontsize',15);
set(0,'defaultaxesfontsize',15);
sph=0.001; % space between each figure
spv=0.5;
spv2=0.01;
mgv=0.05;pdv=0;

t = fig_cell{end};
d = size(fig_cell,2);

for r = floor(sqrt(d)):-1:1
    if mod(d, r)==0
        break;
    end
end
c = d/r;

minI = min(t(:));
maxI = max(t(:));
figure(fn);
fn = fn+1;
colormap gray

for i=1:d
    if size(fig_cell{i},3)>1
        fig_cell{i} = sqrt(sum(fig_cell{i}.^2,3));
    end
end

for i = 1:d
    subaxis(r,c,i, 'Spacing', sph, 'Padding', pdv, 'Margin', mgv);
    imshow(fig_cell{i},[minI maxI]);axis equal tight off;
    title(fig_title{i})
    set(gca,'xtick',[],'ytick',[])
end

% set(gcf,'pos',[ 792         124          210        1263])

linkaxes(findall(0, 'type', 'axes'))
%             print('-dpng','-r600',[num2str(id) '_xy_lr.png'])