
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

%added for calculation for cov and  mean

cd('desk_1/desk');
Filesb = dir(strcat('*.png'));

length(Filesb)
i=2; j=2;
Filesb{i}(j).name;
Filesb(i).name
A=zeros(); map=[];
newCat=[];
for i = 1: length(Filesb)
    %cap25 till cap65
    if i<42
        [A,map] = imread(Filesb(i).name);
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newCAP= [newCat;newVector];
    end
    %76 elements of coffee_mug
    if i>41 && i < 118 
        [A,map] = imread(Filesb(i).name);
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newcoffee_mug= [newcoffee_mug;newVector];
    end
    %elements of soda_can
    if i>117 && i < 186 
        [A,map] = imread(Filesb(i).name);
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newsoda_can= [newsoda_can;newVector];
    end
    
end


%calculation of cov and mean for cap
COV_cap=cov(newCAP);
mu_cap = mean(newsoda_can);

%calculation of cov for coffee_mug
COV_coffee_mug=cov(newcoffee_mug);
mu_coffee = mean(newsoda_can);

%calculation of cov for soda_can
COV_soda_can=cov(newsoda_can);
mu_soda_can = mean(newsoda_can);

%features for GP





