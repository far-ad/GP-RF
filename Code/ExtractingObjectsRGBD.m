clear all;
load('desk_1.mat');
cd('desk_1');

for i = 1:length(bboxes)
    if ~isempty(bboxes{1,i})
        title = strcat('desk_1_',int2str(i),'.png');
        pic = imread(title);
        
        for j = 1:length(bboxes{i})
            xmin    = bboxes{j,i}.left(j);
            ymin    = bboxes{j,i}.top(j);
            width   = bboxes{j,i}.right(j) - bboxes{j,i}.left(j);
            height  = bboxes{j,i}.bottom(j) - bboxes{j,i}.top(j);
            
            subpic = imcrop(pic, [xmin ymin width height]);
            imshow(subpic)
        end
    end
end
cd('..');