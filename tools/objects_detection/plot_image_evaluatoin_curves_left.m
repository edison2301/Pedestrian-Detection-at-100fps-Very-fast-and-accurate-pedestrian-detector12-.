close all;
clear

green = [115 210 22]/255;
orange = [245 121 0]/255;
blue = [52 101 164]/255;
%purple = [117 80 123]/255;
purple = [173 127 168]/255;
red = [204 0 0]/255;
grey = [85 87 83]/255;
show = false;

%range = [1e-2, 1e-0];
%range = [1e-0-0.000001, 1e-0+0.000001];
range = [1e-1, 1e-0];
%range = [1e-2, 1e-1];
%range = [1e-1-0.01, 1e-1+0.01];
figure_h = figure;hold on; grid on
axis([0,.5,0,1])
set(gca,'YTick', [0:.1:1],'XTick',[0:1/16:0.5],'XTickLabel','0|1/16|2/16|3/16|4/16|5/16|6/16|7/16|8/16');
fnt={ 'FontSize',14 };
xlabel('occlusion level',fnt{:});
ylabel('mean recall',fnt{:});
 set(gca,'XDir','Reverse')
resultfolder='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/';

%full baseline
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/baseline/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
plot(crop,meany,'o','color',red, 'markersize', 5, 'markerfacecolor', red,'HandleVisibility','off');

[a,b] = stairs(crop, meany);
a = [a(1); a(1:end-1)];
b = [b(2:end); b(end)];
%baselineArea = plotStairs(figure_h, crop, meany,1, 3, 'k');
s = stairs(a,b,'Color', red,'lineWidth',3);

baselineArea = getArea(crop, meany);
baselineLegend='brute force 100%';

%plot fillup
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/fillup';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
fillupArea = plotStairs(figure_h, crop, meany,baselineArea, 3, green);
fillupLegend=['fill-up ' sprintf('%.0f',fillupArea) '%'];

%plot compound
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/franken_final';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
compoundArea = plotStairs(figure_h, crop, meany,baselineArea, 3, orange);
compoundLegend=['compound ' sprintf('%.0f',compoundArea) '%'];

%baseline 2
aa = [a(1) a(1) a(9) a(9) a(17)];
bb = [b(1) b(9) b(9) b(17) b(17)];
plot([a(1) a(9) a(17)] ,[b(1) b(9) b(17)], 'o','Color', grey,  'markersize', 8, 'markerfacecolor', grey,'HandleVisibility','off')
b2 = plot (aa, bb, 'Color', grey, 'LineStyle', '--','lineWidth',3);
baseline2Area = getArea(aa, bb);
baseline2Legend=['3 classifiers '  sprintf('%.0f',baseline2Area/baselineArea*100) '%'];

%plot biased
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/artificial_franken';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
biasedArea=plotStairs(figure_h, crop, meany,baselineArea, 3, blue)
biasedLegend=['biased '  sprintf('%.0f',biasedArea) '%'];



%naive
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/artificial_normal/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
naiveArea = plotStairs(figure_h, crop, meany,baselineArea, 3, purple);
naiveLegend=['naive '  sprintf('%.0f',naiveArea) '%'];




h_legend = legend(baselineLegend, fillupLegend, compoundLegend,baseline2Legend, biasedLegend , naiveLegend,'Location','SouthEast');
set(h_legend,'FontSize',15);
uistack(s, 'top') 
uistack(b2, 'bottom') 




