# Source Code
  
  ## dataset
     - The directory we use to store the WIDER FACE dataset, as well as training data we prepared for HOG SVM.
       + please download the two rar files hog_svm.rar and wider_face.rar and extract them to this dataset folder, and also keep these two rar files in the folder as well, since we use these rar files for our training notebooks. We don't include these in the submitted files because of their heavy sizes.
       + In hog_svm, there are hard_negatives files that we used for hard negative mining. There are also cropped_positives and cropped_negatives that we initally cropped from WIDER FACE, however our final models don't use them because of their low performance.
       + In processed: where we store the image files as well as the ground truth bounding boxes in csv and json format for convenience
     
     
  ## Demo.ipynb:
      - This is the notebook we use to demo and perform some predictions of our models. The demo images
      are kept in the demo_images directory.

  ## Evaluation.ipynb: 
     - This is the notebook we use to evaluate our models performance.
  
  ## config: the directory we use to save model's training configurations
  
  ## utils.py: where we store some of our helper functions
  
  ## data
       - Directory used to perform augmentation and processing, dataloading for deep learning models
  
  ## model
       - Directory used to store our trained model
  
  ## faster rcnn: 
       - model.py: where we implement the model
       - train.py: where we implement the training code for faster rcnn
       - detect.py:  where we implement the inference function for faster rcnn
       - Train Faster RCNN.ipynb: notebook used to train the model
      - detect Faster RCNN.ipynb:notebook used to do inference on the test set

  ## SSD: 
      - model.py: where we implement the ssdmodel
      - train.py: where we implement the training code for ssd
      - detect.py:  where we implement the inference function for SSD 
      - Train SSD.ipynb: notebook used to train the model
      - SSD detect.ipynb:notebook used to do inference on the test set

  ## svm
     - train hog svm-baseline.inpynb: colab notebook used to train baseline svm model and implement hard negative mining
     - train hog svm-hard negative mining.ipynb: cola b notebook used to train hog svm with hard negative mining
     - svm_detect.py:  where we implement the inference function for hog svm
     - detect SVM.ipynb: notebook used to do inference on the test set
 
  
  ## widerface_evaluate
     - evaluation.py: code to evaluate the results of our model's inferences on the test set
     - predictions: the directory where we store our inference files on the test set

  
  
