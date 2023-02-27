# MATLAB-Deep-Learning
Mirrored digit classification 
(Added 2nd rendition of MATLAB to fix typos [not in the code itself -- in the comments] because I made this program at 3 A.M, and I like grammar.)


Author: Eva Panin, Northeastern University


% REQUIRES DEEP LEARNING TOOLBOX

% The handwritten digits images from MATLAB Deep Lerning (DL) toolbox are utilized.

% These images were originally cropped and centered, reducing the solution space.

% The calssification (identification) of images can be done with a relatively simple network, as demonstrated in MATLAB DL tool examples.

% The program shows that there is tremendeous bias in data with respect to mirrored writting.

% The bias is demonstrated through the simulation of mirrored images, images flipped in left-right direction.

% The program shows that mirrored digits 3 and 5 cannot be identified by the network, trained on the original images. Bias can be removed by data augmentation, which should include image
reflection. Otherwise, regular Affine transformation of original images for network training will improve indentification of mirrored digits besides the mirrored 5.

% The specific data augmentation can be useful in handwriting conversion into digital format. Since mirrored writing is typical in children and brain-damaged adults (Parkinson's disease, tremors, spino-cerebellular degredation), this program can be helpful in grading of tests, scanning documents, etc.  Also, mirrored handwriting is an early sign of mental decline in eldery adults. 
Software analysis of their handwriting can be usful in prediction of aging outcomes.

% The program consists of three steps:

% First, it shows that a trained network can classify images from the original data, serving as test images, which were not present in the
 training set. I removed 100 images from the original data (the digit 3), since this is the most common mirrored image.
 
%Second, the test mirrored images were created from the original images by  horizontal reflections. The nework was trained on full original data (non-mirrored images, all 500 images of 3 are included). 
Classification is shown for all mirrored digits from 0 to 9. The distribution of misclassification is graphed. 
Exmaples of the images and corresponding mis-labels are provided.
The network cannot correctly classify mirror digits 3 and 5.

%The last network is trained again on the original data set and data
 augmentation by rotation and reflection. Classification is shown for all
mirror digits from 0 to 9 for this newly trained network.
MATLAB augmentation tool is used (reflect + flip images). 
 Augmentation is performed during training, so the user does not need to create augmented images before training. 
