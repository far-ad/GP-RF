
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

cd('resizeobjects');
Filesb = dir(strcat('*.png'));





A=zeros(); map=[]; newCAP=[];
newCat=[]; newcoffee_mug=[]; newsoda_soda_can=[]; newsoda_bowl=[]; newsoda_flashlight=[];
for i = 1: length(Filesb)
    %cap
    if i<43 && i>1
        [A,map] = imread(char(Filesb(i,1).name));
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newCAP= [newCAP,newVector];
    end
    %76 elements of coffee_mug
    if i>42 && i < 119 
        [A,map] = imread(char(Filesb(i,1).name));
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newcoffee_mug= [newcoffee_mug,newVector];
    end
    %elements of soda_can
    if i>118 && i < 187 
        [A,map] = imread(char(Filesb(i,1).name));
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newsoda_soda_can= [newsoda_soda_can,newVector];
    end
    %bowl
    if i>186 && i < 297
        [A,map] = imread(char(Filesb(i,1).name));
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newsoda_bowl= [newsoda_bowl,newVector];
    end
    %flashlight
    if i>296 && i < 427
        [A,map] = imread(char(Filesb(i,1).name));
        newVector=reshape(A,[size(A,1)*size(A,2)*size(A,3), 1]);
        newsoda_flashlight= [newsoda_flashlight,newVector];
    end
    
end


%calculation of cov and mean for cap
COV_cap=cov(double(newCAP));
mu_cap = mean(newCAP);

%calculation of cov for coffee_mug
COV_coffee_mug=cov(double(newcoffee_mug));
mu_coffee = mean(newcoffee_mug);

%calculation of cov for soda_can
COV_soda_can=cov(double(newsoda_soda_can));
mu_soda_can = mean(newsoda_soda_can);



%calculation of cov for newsoda_bowl
COV_soda_bowl=cov(double(newsoda_bowl));
mu_soda_bowl = mean(newsoda_bowl);



%calculation of cov for newsoda_bowl
COV_soda_flashlight=cov(double(newsoda_flashlight));
mu_soda_flashlight = mean(newsoda_flashlight);


%features for GP
