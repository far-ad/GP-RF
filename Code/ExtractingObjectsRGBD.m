clear all;
load('table_small_2.mat');
cd('table_small_2');

for i = 1:length(bboxes)
    if ~isempty(bboxes{1,i})
        title = strcat('table_small_2_',int2str(i),'.png');
        pic = imread(title);
        
        for j = 1:length(bboxes{i})
            xmin    = bboxes{i}(j).left;
            ymin    = bboxes{i}(j).top;
            width   = bboxes{i}(j).right - bboxes{i}(j).left;
            height  = bboxes{i}(j).bottom - bboxes{i}(j).top;
            
            label    = bboxes{i}(j).category;
            subpic = imcrop(pic, [xmin ymin width height]);
            
            cd('table_small');
            
            subpic = rgb2gray(subpic);
            
            imwrite(subpic,[label int2str(i) '.png']);
            cd('..');
            
        end
    end
end
cd('..');