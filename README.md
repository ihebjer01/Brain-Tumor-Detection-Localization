# Brain-Tumor-Detection-Localization
We utilized transfer learning with pretrained ResNet50 to predict whether a patient has cancer or not and ResUnet model to locate the tumor based scans or MRIs.
Here is a general outline of the steps involved in this project : 
Data pre-processing: Preprocess the medical images to prepare them for modeling. This includes normalizing pixel values, and augmenting the data to increase the size of the training set. Then we Utilized Transfer learning  with a pretrained ResNet50 model to train a binary classifier that predicts whether a patient has cancer or not
and ResUnet model to train a segmentation model that can localize the tumor within the medical image And finally the evaluation of boths models
