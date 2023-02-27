function MirrorHandWriting
% Author: Eva Panin, Northeastern University
% REQUIRES DEEP LEARNING TOOLBOX
% The handwritten digits images from MATLAB Deep Lerning (DL) toolbox are
% utilized.
% These images were originally cropped and centered, reducing the solution
% space.
% The classification (identification) of images can be done with a relatively
% simple network, as demonstrated in MATLAB DL tool examples.
% The program shows that there is tremendeous bias in data with respect to
% mirrored writting.
% The bias is demonstrated through the simulation of mirrored images, 
% images flipped in left-right direction.
% The program shows that mirrored digits 3 and 5 cannot be identified by
% the network, trained on the original images.
% Bias can be removed by data augmentation, which should include image
% reflection. Otherwise, regular Affine transformation of original images for network training will improve indentification of mirrored 3,
% but not mirrored 5.
% The specific data augmentation can be useful in handwriting conversion
% into digital format. Since mirrored writing is typical in children and
% brain-damaged adults (Parkinson's disease, tremors, spino-cerebellular degredation), 
% this program can be helpful in grading of tests, scanning documents, etc.  
% Also, mirrored handwriting is an early sign of mental decline in eldery adults. 
% Software analysis of their handwriting can be usful in prediction of aging
% outcomes.

% The program consists of three steps:
% First, it shows that a trained network can classify images from the
% original data, serving as test images, which were not present in the
% training set. I removed 100 images from the original data (the digit 3),
% since this is the most common mirrored image.
% Second, the test mirrored images were created from the original images by 
% horizontal reflections. The nework was trained on full original data
% (non-mirrored images, all 500 images of 3 are included). 
% Classification is shown for all mirrored digits from 0 to 9. The distribution of misclassification is graphed. 
% Exmaples of the images and corresponding mis-labels are provided.
% The network cannot correctly classify mirror digits 3 and 5.
% The last network is trained again on the original data set and data
% augmentation by rotation and reflection. Classification is shown for all
% mirror digits from 0 to 9 for this newly trained network.
% MATLAB augmentation tool is used (reflect + flip images). 
% Augmentation is performed during training, so the user does not need to create augmented images before training. 



close all;
clear;
clc;

delete(findall(groot,'Type','figure'));


PAUSEDURATION=5;

fprintf(['This program explores MATLAB hand-written labeled digits images 0-9, 500 instances of each \n', ...
    'The images were originally carefully cropped and centered, reducing solution space. Nevertheless, each digit appeared to be rotated, representing natural writing habbits \n']);
fprintf('The first task is to show that the simple Deep Learning network can succesfully learn digits, what is prediction of digit label for given image \n');

% Load digit images using MATLAB provided load function


% Each image is 28x28 with one channel, grayscale intensity
% There are digits 0-9, each digit has 500 instances 

[Digits, Labels] = digitTrain4DArrayData; %use MATLAB DL function to load images
Nsize=28;
Ndigit=500;
Nlabel=10; % labels are 0-9 digits
Ntest=100;
ImageSize = [Nsize Nsize 1];

fprintf('Display some of images \n');
figure(1);
set(gcf,'Position',[715,306,560,535]);
for i=1:9
    ind=randsample(Ndigit*Nlabel,1);
    subplot(3,3,i);
    imagesc(Digits(:,:,1,ind));
    title('Label: '+string(Labels(ind)));
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    colormap gray;
end
sgtitle('Example of digit images');
drawnow;

pause(PAUSEDURATION);
close(1);

fprintf('100 images of digit 3 served as test images. Train network on the rest of images. Let network identify labels of these test images \n');
%Define indices of images for digit 3
D3I=Labels=='3';
%Index of first Ntest images of 3
D3Itest=D3I;
ind=find(D3Itest);
D3Itest(ind(Ntest)+1:end)=0;
%Remove test images of 3 from training set
Dtrain = Digits;
Dtrain(:,:,:,D3Itest)=[];
Ltrain = Labels;
Ltrain(D3Itest)=[];
Dtest = Digits(:,:,:,D3Itest);

% Create classification network according to MATLAB recipe
layers = [ ...
    imageInputLayer(ImageSize,'Name', 'Image input layer')
    convolution2dLayer(5,20,'Name','2D convolution of 5x5')
    reluLayer('Name','ReLu')
    maxPooling2dLayer(2,'Stride',2,'Name','Max pooling by factor 2')
    fullyConnectedLayer(Nlabel,'Name','Fully connected layer')
    softmaxLayer('Name','Soft max layer')
    classificationLayer('Name','Classification layer')];

%Visualize network structure
figure(1);
lgraph=layerGraph(layers);
plot(lgraph);
title('Network');
drawnow;

% Train!
options = trainingOptions('adam', 'MaxEpochs',10,'InitialLearnRate',3e-3, 'Verbose',false, 'Plots','training-progress');
net = trainNetwork(Dtrain,Ltrain,layers,options);
pause(3);
close(1);
delete(findall(groot,'Type','figure'));
commandwindow;
%Predict labels of test images
Lpredict_test = classify(net,Dtest);
fprintf('Precision if classifying test images of digit 3 is %6.2f %%\n', 100*sum(Lpredict_test=='3')/Ntest);
fprintf('Precision is quite high for images outside training set \n');
fprintf('Example of mis-identified images \n');
%Example of mis-identified image
ind_mis=find(Lpredict_test~='3');
figure(1);
imagesc(Dtest(:,:,1,ind_mis(1)));
title('Incorrect Label: '+string(Lpredict_test(ind_mis(1))));
set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
colormap gray;
drawnow;

pause(PAUSEDURATION);
close(1);


fprintf('Now mirror images of certain digits serve as test images. Train network is trained on all original images. Network will try to predict label of mirror images \n');

% Create classification network according to MATLAB receipe
layers = [ ...
    imageInputLayer(ImageSize, 'Name', 'Image input layer')
    convolution2dLayer(5,20,'Name','2D convolution of 5x5')
    reluLayer('Name','ReLu')
    maxPooling2dLayer(2,'Stride',2,'Name','Max pooling by factor 2')
    fullyConnectedLayer(Nlabel,'Name','Fully connected layer')
    softmaxLayer('Name','Soft max layer')
    classificationLayer('Name','Classification layer')];

%Visualize network structure
figure(1);
lgraph=layerGraph(layers);
plot(lgraph);
title('Network');
drawnow;

%Options in training
options = trainingOptions('adam', 'MaxEpochs',20,'InitialLearnRate',3e-3, 'Verbose',false, 'Plots','training-progress');
% Train!
net = trainNetwork(Digits,Labels,layers,options);
pause(3);
close(1);
delete(findall(groot,'Type','figure'));
commandwindow;

%Loop of all digits for prediction
for d=0:9
    
    %Extract images of digit
    Dtest=Digits(:,:,:,Labels==num2str(d));
    Lprediction_Dtest = classify(net,Dtest);
    fprintf('Precision in classifying digit %u from training set is %6.2f %% \n', d, 100*sum(Lprediction_Dtest==num2str(d))/Ndigit);
    %Predict label for mirror digit
    %Create mirror handwriting of digit
    Dtestmirror=flip(Dtest,2);
    %Predict label for mirror digit
    Lprediction_Dtestmirror = classify(net,Dtestmirror);
    fprintf('Precision in classifying mirror digit %u  is %6.2f %% \n', d, 100*sum(Lprediction_Dtestmirror==num2str(d))/Ndigit);

    figure(1);
    set(gcf,'Position',[1,494,368,300]);
    histogram(Lprediction_Dtestmirror);
    title(['Classification of mirror ',num2str(d)]);
    drawnow;

    %Show examples of training set and mirror images
    figure(2);
    set(gcf,'Position',[377,468,1058,354]);
    Nsample=10;
    Drandi = randsample(Ndigit,Nsample);
    for i=1:Nsample
        subplot(2,Nsample,i);
        imagesc(Dtest(:,:,1,Drandi(i)));
        title('Prediction '+string(Lprediction_Dtest(i)));
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        colormap gray;
        subplot(2,Nsample,i+Nsample);
        imagesc(Dtestmirror(:,:,1,Drandi(i)));
        title('Prediction '+string(Lprediction_Dtestmirror(i)));
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
        colormap gray;
    end
    drawnow;

    pause(PAUSEDURATION);
end
fprintf('\n');
fprintf('Network can not identify mirror 3 and 5 at all. \n');
fprintf('For example, mirror 3 was identified as 8, 2 or 6. \n');
fprintf('Therefore, there is significant data set bias w.r.t. mirror handwriting. \n')
fprintf('One can attempt to remove bias in data by creating (simulating) larger data space. This is achieved through data augmentation by significant rotation and image reflection. \n');
fprintf('Not shown here explicitly, but image rotation alone will improve clasification of mirror 3 but not mirror 5 \n');

pause(PAUSEDURATION);
close(1);
close(2);

% Augmentation by significant rotation
% MATLAB augmentation is performed during training process through
% imageAugmenter object
% Since solution space is larger, more parameters network can be used

Imgtrain = augmentedImageDatastore(ImageSize,Digits,Labels,'DataAugmentation',imageDataAugmenter('RandRotation',[-90,90],'RandXReflection',true));
layers = [
    imageInputLayer(ImageSize,'Name', 'Image input layer')
    
    convolution2dLayer(3,8,'Padding','same','Name','2D convolution of 3x3 1')
    batchNormalizationLayer('Name','Batch Normalization 1')
    reluLayer('Name','ReLu 1')  
    %tanhLayer('Name','Tanh activation 1')
    maxPooling2dLayer(2,'Stride',2,'Name','Max pooling by 2 1')
    
    convolution2dLayer(3,16,'Padding','same','Name','2D convolution of 3x3 2')
    batchNormalizationLayer('Name','Batch Normalization 2')
    reluLayer('Name','ReLu 2') 
    %tanhLayer('Name','Tanh activation 2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Max pooling by 2 2')
    
    convolution2dLayer(3,32,'Padding','same','Name','2D convolution of 3x3 3')
    batchNormalizationLayer('Name','Batch Normalization 3')
    reluLayer('Name','ReLu 3')
    %tanhLayer('Name','Tanh activation 3')
    
    fullyConnectedLayer(Nlabel,'Name','Fully connected layer')
    softmaxLayer('Name','Soft Max Layer')
    classificationLayer('Name','Classification Layer')];
figure(1);
lgraph=layerGraph(layers);
plot(lgraph);
title('Network');
drawnow;
pause(5);
options = trainingOptions('adam','MaxEpochs',50,'Shuffle','every-epoch','Plots','training-progress','Verbose',false);
net = trainNetwork(Imgtrain,layers,options);
pause(3);
close(1);


fprintf('\n');
%Loop of all digits for prediction
figure(1);
set(gcf,'Position',[10,80,1880,910]);
for d=0:9
    
    %Extract images of digit
    Dtest=Digits(:,:,:,Labels==num2str(d));
    Lprediction_Dtest = classify(net,Dtest);
    fprintf('After data augmentation, precision in classifying digit %u from unmirrored images %6.2f %% \n', d, 100*sum(Lprediction_Dtest==num2str(d))/Ndigit);
    %Predict label for mirror digit
    %Create mirror handwriting of digit
    Dtestmirror=flip(Dtest,2);
    %Predict label for mirror digit
    Lprediction_Dtestmirror = classify(net,Dtestmirror);
    fprintf('After data augmentation, precision in classifying mirror digit %u  is %6.2f %% \n', d, 100*sum(Lprediction_Dtestmirror==num2str(d))/Ndigit);

    subplot(2,5,d+1);
    histogram(Lprediction_Dtestmirror);
    title(['Classification of mirror ',num2str(d)]);
    drawnow;
end
sgtitle('Classification after data augmentation');


end