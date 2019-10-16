# Indian-Sign-Language-Alphabets-Translation

Created a dataset of gesture of each alphabet in Indian Sign Language, used resnet32 architecture for creating a classifier and opencv for separating user's hand from the background and classifying each frame.

For creating the dataset multiple videos of each gesture were captured and a set number of frames were exrtracted for each gesture.

Then I trained a resnet32 classifier on google colab cloud and exported the model to local cpu for inference. In order to get user's skin color, nine small boxes are displayed and user has to cover the boxes with their hand. In each frame the gesture is extracted from the background and classified and if an alphabet is classified for last 20/30 frames that alphabet is printed.


The datasets can be found at 
https://drive.google.com/drive/folders/1pr8kSD85nk-tIzo9G5NzYpWen8DsMiaO?usp=sharing
https://drive.google.com/drive/folders/1ddQA2rhL3Aq39XOsjZOZMMq3Jufjn2WC?usp=sharing

The colab link is:
https://colab.research.google.com/drive/1EN6GG1Sm0uD5_DE5BNY-YAenuYjTJmIV
