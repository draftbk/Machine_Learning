
load('Data-Ass2.mat');
lern = data(1:2,1:2000)';
test = data(1:2,2001:3000)';
lernLabel = data(3,1:2000);
testLable = data(3,2001:3000);

svmModel = svmtrain(lern, lernLabel,'kernel_function','rbf','showplot',true);
predict_label = svmclassify(svmModel,test,'showplot',true); 