clear all;
load('desk_1.mat');
cd('desk_1');

E = [];

for i = 1:length(bboxes)
    if ~isempty(bboxes{1,i})
        title = strcat('desk_1_',int2str(i),'.png');
        pic = imread(title);
        
        for j = 1:length(bboxes{i})
            xmin    = bboxes{i}(j).left;
            ymin    = bboxes{i}(j).top;
            width   = bboxes{i}(j).right - bboxes{i}(j).left;
            height  = bboxes{i}(j).bottom - bboxes{i}(j).top;
            
            label    = bboxes{i}(j).category;
            subpic = imcrop(pic, [xmin ymin width height]);
            
            cd('desk');
            
            imwrite(subpic,[label int2str(i) '.png']);
            cd('..');
            
        end
    end
end
cd('..');