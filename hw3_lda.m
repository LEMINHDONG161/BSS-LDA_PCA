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

test=read_data_test(:,1:2048)';
train=read_data_train(:,1:2048)';
c=1000;
y=read_data_train(:,2049)'; % class labels

% Scatter matrix computation
[Sw,Sb,Sm]=scatter_mat(train,y);

% Eigendecomposition
[V,D]=eig(inv(Sw)*Sb);

% Sort the eigenvalues in descending order and rearrange the eigenvectors accordingly
s=diag(D);
[s,ind]=sort(s,1,'descend');
V=V(:,ind);
% Select in A the eigenvectors corresponding to non-zero eigenvalues
A=V(:,1:c-1);
% Project the data set on the space spanned by the column vectors of A
Y_test=A'*test;
Y_train=A'*train;
% labeled:Class 1

        class1_data= read_data_train(find(read_data_train(:,2049) == 1),1:2048);
        Y_c1=A'*class1_data';
        [m1_hat, S1_hat]=Gaussian_ML_estimate(Y_c1);
        p1= length(find(read_data_train(:,2049)==1))/32036;

% labeled:Class 2
         class2_data= read_data_train(find(read_data_train(:,2049) == 2),1:2048);
         Y_c2=A'*class2_data';
         [m2_hat, S2_hat]=Gaussian_ML_estimate(Y_c2);
         p2= length(find(read_data_train(:,2049)==2))/32036;
% labeled:Class 3
       class3_data= read_data_train(find(read_data_train(:,2049) == 3),1:2048);
        Y_c3=A'*class3_data';
        [m3_hat, S3_hat]=Gaussian_ML_estimate(Y_c3);
        p3= length(find(read_data_train(:,2049)==3))/32036;
% labeled:Class 4
      class4_data= read_data_train(find(read_data_train(:,2049) == 4),1:2048);
      Y_c4=A'*class4_data';
      [m4_hat, S4_hat]=Gaussian_ML_estimate(Y_c4);
      p4= length(find(read_data_train(:,2049)==4))/32036;
% labeled:Class 5
      class5_data= read_data_train(find(read_data_train(:,2049) == 5),1:2048);
      Y_c5=A'*class5_data';
      [m5_hat, S5_hat]=Gaussian_ML_estimate(Y_c5);
      p5= length(find(read_data_train(:,2049)==5))/32036;
% labeled:Class 6
      class6_data= read_data_train(find(read_data_train(:,2049) == 6),1:2048);
      Y_c6=A'*class6_data';
      [m6_hat, S6_hat]=Gaussian_ML_estimate(Y_c6);
      p6= length(find(read_data_train(:,2049)==6))/32036;
% labeled:Class 7
     class7_data= read_data_train(find(read_data_train(:,2049) == 7),1:2048);
     Y_c7=A'*class7_data';
       [m7_hat, S7_hat]=Gaussian_ML_estimate(Y_c7);
       p7= length(find(read_data_train(:,2049)==7))/32036;
% labeled:Class 8
       class8_data= read_data_train(find(read_data_train(:,2049) == 8),1:2048);
       Y_c8=A'*class8_data';
       [m8_hat, S8_hat]=Gaussian_ML_estimate(Y_c8);
       p8= length(find(read_data_train(:,2049)==8))/32036;
% labeled:Class 9
    class9_data= read_data_train(find(read_data_train(:,2049) == 9),1:2048);
    Y_c9=A'*class9_data';
    [m9_hat, S9_hat]=Gaussian_ML_estimate(Y_c9);
    p9= length(find(read_data_train(:,2049)==9))/32036;
% labeled:Class 10
   class10_data= read_data_train(find(read_data_train(:,2049) == 10),1:2048);
   Y_c10=A'*class10_data';
   [m10_hat, S10_hat]=Gaussian_ML_estimate(Y_c10);
   p10= length(find(read_data_train(:,2049)==10))/32036;
   
 %m_hat and S_hat and p
 m_hat=[m1_hat m2_hat m3_hat m4_hat m5_hat m6_hat m7_hat m8_hat m9_hat m10_hat];
 S_hat=(1/10)*(S1_hat+S2_hat+S3_hat+S4_hat+S5_hat+S6_hat+S7_hat+S8_hat+S9_hat+S10_hat);
 
 p=[p1 p2 p3 p4 p5 p6 p7 p8 p9 p10]';
 %p=[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1];
 S(:,:,1)=S_hat;  S(:,:,2)=S_hat; S(:,:,3)=S_hat; S(:,:,4)=S_hat; S(:,:,5)=S_hat; 
 S(:,:,6)=S_hat;  S(:,:,7)=S_hat; S(:,:,8)=S_hat; S(:,:,9)=S_hat; S(:,:,10)=S_hat; 



% Euclidean distance classifier
%z_euclidean=euclidean_classifier(m_hat,Y_test);

%  Mahalanobis distance classifier
z_mahalanobis=mahalanobis_classifier(m_hat,S_hat,Y_test);

%  bayes classifier and provide as input the matrices

%z_bayesian=bayes_classifier(m_hat,S,p,Y_test);
%% Error calculation

%err_euclidean = (1-length(find(read_data_test(:,2049)==z_euclidean'))/8009)
err_mahalanobis = (1-length(find(read_data_test(:,2049)==z_mahalanobis'))/8009)
%err_bayesian = (1-length(find(read_data_test(:,2049)==z_bayesian'))/8009)




