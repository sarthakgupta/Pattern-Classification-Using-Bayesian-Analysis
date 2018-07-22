A1=imread('Test_digit.png');
RGB = rgb2gray(A1);
I1=reshape(RGB,1,[]);
save('matlab','I1');

[m,n]=size(RGB);

I2=zeros(1,m*n);
start=1;

for i=1:m
    I2(1,start:n*i)=RGB(i,:);
    start=start+m;
end
 
save('matlab','I2');
% filename='../../../Data Sets/mnist_train_60000.csv';
% A  = csvread(filename,0,1);
% B  = csvread(filename,0,0);
% I=A(85,:);
% Im_1=reshape(I,[28,28]);
% Im_1=Im_1.';
% imshow(Im_1)
