close all;
clear
loadColors;
%range = [1e-2, 1e-0];
%range = [1e-0-0.000001, 1e-0+0.000001];
%range = [1e-2, 1e-1];
%range = [1e-1-0.01, 1e-1+0.01];
%range =[1e-4, 1]

range = [1e-1, 1e-0];
show = true;
resultfolder='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/';

%%setup figure
figure_h = figure;hold on; grid on
axis([0,.5,0,1])
set(gca,'XTick',[0:1/16:0.5],'XTickLabel','0|2/32|4/32|6/32|8/32|10/32|12/32|14/32|16/32');
fnt={ 'FontSize',14 };
xlabel('occlusion level',fnt{:});
ylabel('mean recall',fnt{:});
set(gca,'XDir','Reverse')




%full baseline
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/baseline/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
plot(crop,meany,'o','color',red, 'markersize', 5, 'markerfacecolor', red,'HandleVisibility','off');

[a,b] = stairs(crop, meany);
a = [a(1); a(1:end-1)];
b = [b(2:end); b(end)];
%baselineArea = plotStairs(figure_h, crop, meany,1, 3, 'k');
s = stairs(a,b,'Color', red,'lineWidth',3);

baselineArea = getArea(crop, meany);
%baselineArea = getArea(a, b);
baselineLegend='brute force 100%';



    

%plot fillup classifier
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_non_recursive';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
fillupArea = plotStairs(figure_h, crop, meany,baselineArea, 3, green);
fillupLegend=['fill-up ' sprintf('%.0f',fillupArea) '%'];

%plot compound classifier
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_recursive';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
compoundArea = plotStairs(figure_h, crop, meany,baselineArea, 3, orange);
compoundLegend=['compound ' sprintf('%.0f',compoundArea) '%'];


%baseline 2
aa = [a(1) a(1) a(17) a(17) a(33)];
bb = [b(1) b(17) b(17) b(33) b(33)];
plot([a(1) a(17) a(33)] ,[b(1) b(17) b(33)], 'o','Color', grey,  'markersize', 8, 'markerfacecolor', grey,'HandleVisibility','off')
b2 = plot (aa, bb, 'Color', grey, 'LineStyle', '--','lineWidth',3);
baseline2Area = getArea(aa, bb);
baseline2Legend=['3 classifiers '  sprintf('%.0f',baseline2Area/baselineArea*100) '%'];



%plot biased 
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_art';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
biasedArea=plotStairs(figure_h, crop, meany,baselineArea, 3, blue)
biasedLegend=['biased '  sprintf('%.0f',biasedArea) '%'];



%naive
show = true;
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/full_art/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
naiveArea = plotStairs(figure_h, crop, meany,baselineArea, 3, purple);
naiveLegend=['naive '  sprintf('%.0f',naiveArea) '%'];

%baseline 4
aa = [a(1) a(1) a(7) a(7) a(17) a(17) a(25) a(25) a(33)];
bb = [b(1) b(7) b(7) b(17) b(17) b(25) b(25) b(33) b(33)];
%plot (aa, bb, 'k--','lineWidth',3);
baseline4Area = getArea(aa, bb);
baseline4Legend=['basline 4 '  sprintf('%.0f',baseline4Area/baselineArea*100) '%']










h_legend = legend(baselineLegend, fillupLegend, compoundLegend,baseline2Legend, biasedLegend , naiveLegend,'Location','SouthEast');
set(h_legend,'FontSize',15);
uistack(s, 'top') 
uistack(b2, 'bottom') 