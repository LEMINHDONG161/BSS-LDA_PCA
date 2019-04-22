close all;
clear all
%% Load Data File
load('C:\Users\jys\Desktop\Artificial Inteligence\AI\new_data10'); 
read_data_train = data_train;
read_data_test = data_test;
% Result
%read_data_train 32036 x2049 ; 
%read_data_test  8009x2049;
% 2049 column is the class number of 1 to 10
%% Train Data by Using Data_Train and by Assuming That It Is Gaussian Distribution
% step 1: extract classes from data_train
% step 2: calculate the mean values and covariance matrix for each class using Gaussian_ML_estimate
% step 3: base on train data calculate the priori probabilities of each class. They are also parameters for beyansian classifier


m=1000;
% labeled:Class 1

        class1_data= read_data_train(find(read_data_train(:,2049) == 1),1:2048);
        [eigenval_c1,eigenvec_c1,explained_c1,Y_c1,mean_vec_c1]=pca_fun(class1_data',m);
        [m1_hat, S1_hat]=Gaussian_ML_estimate(Y_c1);
        p1= length(find(read_data_train(:,2049)==1))/32036;

% labeled:Class 2
       class2_data= read_data_train(find(read_data_train(:,2049) == 2),1:2048);
         
        [eigenval_c2,eigenvec_c2,explained_c2,Y_c2,mean_vec_c2]=pca_fun(class2_data',m);
        [m2_hat, S2_hat]=Gaussian_ML_estimate(Y_c2);
         p2= length(find(read_data_train(:,2049)==2))/32036;
% labeled:Class 3
       class3_data= read_data_train(find(read_data_train(:,2049) == 3),1:2048);
       
        [eigenval_c3,eigenvec_c3,explained_c3,Y_c3,mean_vec_c3]=pca_fun(class3_data',m);
        [m3_hat, S3_hat]=Gaussian_ML_estimate(Y_c3);
        p3= length(find(read_data_train(:,2049)==3))/32036;
% labeled:Class 4
      class4_data= read_data_train(find(read_data_train(:,2049) == 4),1:2048);
      
        [eigenval_c4,eigenvec_c4,explained_c4,Y_c4,mean_vec_c4]=pca_fun(class4_data',m);
        [m4_hat, S4_hat]=Gaussian_ML_estimate(Y_c4);
      p4= length(find(read_data_train(:,2049)==4))/32036;
% labeled:Class 5
      class5_data= read_data_train(find(read_data_train(:,2049) == 5),1:2048);
     
        [eigenval_c5,eigenvec_c5,explained_c5,Y_c5,mean_vec_c5]=pca_fun(class5_data',m);
        [m5_hat, S5_hat]=Gaussian_ML_estimate(Y_c5);
        p5= length(find(read_data_train(:,2049)==5))/32036;
% labeled:Class 6
      class6_data= read_data_train(find(read_data_train(:,2049) == 6),1:2048);
   
        [eigenval_c6,eigenvec_c6,explained_c6,Y_c6,mean_vec_c6]=pca_fun(class6_data',m);
        [m6_hat, S6_hat]=Gaussian_ML_estimate(Y_c6);
      p6= length(find(read_data_train(:,2049)==6))/32036;
% labeled:Class 7
     class7_data= read_data_train(find(read_data_train(:,2049) == 7),1:2048);
     
        [eigenval_c7,eigenvec_c7,explained_c7,Y_c7,mean_vec_c7]=pca_fun(class7_data',m);
        [m7_hat, S7_hat]=Gaussian_ML_estimate(Y_c7);
       p7= length(find(read_data_train(:,2049)==7))/32036;
% labeled:Class 8
     class8_data= read_data_train(find(read_data_train(:,2049) == 8),1:2048);
      
        [eigenval_c8,eigenvec_c8,explained_c8,Y_c8,mean_vec_c8]=pca_fun(class8_data',m);
        [m8_hat, S8_hat]=Gaussian_ML_estimate(Y_c8);
       p8= length(find(read_data_train(:,2049)==8))/32036;
% labeled:Class 9
    class9_data= read_data_train(find(read_data_train(:,2049) == 9),1:2048);
   
        [eigenval_c9,eigenvec_c9,explained_c9,Y_c9,mean_vec_c9]=pca_fun(class9_data',m);
        [m9_hat, S9_hat]=Gaussian_ML_estimate(Y_c9);
        p9= length(find(read_data_train(:,2049)==9))/32036;
% labeled:Class 10
   class10_data= read_data_train(find(read_data_train(:,2049) == 10),1:2048);
   
        [eigenval_c10,eigenvec_c10,explained_c10,Y_c10,mean_vec_c10]=pca_fun(class10_data',m);
        [m10_hat, S10_hat]=Gaussian_ML_estimate(Y_c10);
   p10= length(find(read_data_train(:,2049)==10))/32036;
   
 %m_hat and S_hat and p
 m_hat=[m1_hat m2_hat m3_hat m4_hat m5_hat m6_hat m7_hat m8_hat m9_hat m10_hat];
 S_hat=(1/10)*(S1_hat+S2_hat+S3_hat+S4_hat+S5_hat+S6_hat+S7_hat+S8_hat+S9_hat+S10_hat);
 
 p=[p1 p2 p3 p4 p5 p6 p7 p8 p9 p10]';
 %p=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];
 S(:,:,1)=S_hat;  S(:,:,2)=S_hat; S(:,:,3)=S_hat; S(:,:,4)=S_hat; S(:,:,5)=S_hat; 
 S(:,:,6)=S_hat;  S(:,:,7)=S_hat; S(:,:,8)=S_hat; S(:,:,9)=S_hat; S(:,:,10)=S_hat; 
 
 
 
   
%% TEST DATA by Using data_test. Use 3 distance measure Euclidean Mahalanobis and Baysian and compare three results

test=read_data_test(:,1:2048);

[eigenval_test,eigenvec_test,explained_test,Y_test,mean_vec_test]=pca_fun(test',m);
% Euclidean distance classifier
z_euclidean=euclidean_classifier(m_hat,Y_test);

%  Mahalanobis distance classifier
%z_mahalanobis=mahalanobis_classifier(m_hat,S_hat,Y_test);

%  bayes classifier and provide as input the matrices

%z_bayesian=bayes_classifier(m_hat,S,p,test');
%% Error calculation

err_euclidean = (1-length(find(read_data_test(:,2049)==z_euclidean'))/8009)
%err_mahalanobis = (1-length(find(read_data_test(:,2049)==z_mahalanobis'))/8009)
%err_bayesian = (1-length(find(read_data_test(:,2049)==z_bayesian'))/8009)




