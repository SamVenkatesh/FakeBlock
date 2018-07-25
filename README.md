# FakeBlock

## What
A Deep Learning application built using Python to recognize emotion from facial expressions.

#### TLDR/Usage
First, [install Keras](https://keras.io/#installation) along with [TensorFlow](https://www.tensorflow.org/install/install_mac), you could simply run `pip install keras` and `pip install tensorflow` respectively if you have PyPI. You'll also need [OpenCV](https://pypi.org/project/opencv-python/) or `pip install opencv-python`.  

To run the application, simply execute `python3 WebCam.py` or your respective command to run python3. To train the network with different parameters, you would have to make minor modifications to `BuildTrainTestCNN.py`.  
  
I am working on adding functionality to make this process easier. For the time being, you could tweak the default values for training/testing parameters in the aforementioned file.  
The values I've used are by no means perfect. I encourage you to find better configurations.

## Why
I recently completed the CSCI 567 - Machine Learning Course at USC and I enjoyed working on projects throughout that class. 
So, I decided to work on a side project over the summer just to see what I could build with what I had learned.

I came across an [old contest on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and I was intrigued so I started working on it.

## How
I decided to use a [Convolutional Neural Network](http://cs231n.github.io/convolutional-networks/) (CNN) for this project. Neural Networks were always a favorite of mine (I'm not entirely sure why I'm partial to Neural Nets, but they've always seemed very intuitive to me).  
Additionally, in order to retrieve the user's face, I used [OpenCV](https://opencv.org/) to open a continuous feed from the WebCam and it's [Haar Cascade Classifier](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html) to detect faces from the resulting frames.
Once I have this face, my CNN can do it's thing.

#### Side Note:
For those of you who aren't really familiar with Neural Networks or are not interested in diving deep on them, [this video](https://www.youtube.com/watch?v=aircAruvnKk) by 3Blue1Brown is an excellent introduction to the topic and I highly recommend it to everyone, regardless of prior knowledge.

#### Back on topic:
Additionally, for my course, I had to design and implement a Neural Network from scratch to classify the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) so it seemed fitting that I could use a CNN for other image processing problems as well.

Luckily for me, the aforementioned Kaggle contest contained a fairly comprehensive dataset of people making faces and a corresponding label of the emotion being displayed in the image.  
Specifically, it contains 28,709 images labelled with one of seven emotions - Neutral, Happy, Sad, Angry, Surprised, Fearful and Disgusted.   
Despite my limited experience with Machine Learning, I have learned that the 80-20 rule applies here as well. 80% of the job is getting/filtering/sorting/labelling data. The actual "learning" part is not overly complicated.


<p align="center">
  <img src ="https://imgs.xkcd.com/comics/machine_learning.png" />
</p>

Credit-[Randall Munroe's xkcd](https://xkcd.com/1425)

Designing my network itself was not as difficult as I had anticipated. I referred to [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) and tried to follow their example of applying Convolution and Max-Pooling layers at various stages.  
  
Ultimately, I ended up with the following network.

<p align="center">
  <img src ="https://drive.google.com/uc?id=1PAP15NnHnsPqW2Il4RmZCJD09Kq4CkQY" />
</p>
  
For the implementation, I chose [Keras](https://keras.io/) with a [TensorFlow backend](https://www.tensorflow.org/guide/keras). Keras acts like a wrapper over TensorFlow and at the time, it seemed easier to get started with.  
In hindsight, performing the same in TensorFlow would not have been any different, save for changes in syntax. TensorFlow's [tutorials](https://www.tensorflow.org/tutorials/) and [documentation](https://www.tensorflow.org/api_docs/) are excellent.

## Training
I trained the CNN for 200 epochs with a random 80-20 train-test split of the dataset with the training data being shuffled at the start of each iteration. This achieved a Mean Square Error (MSE) of 0.068.  
   
I'm currently debating training this on one of [Amazon's Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) on AWS for 1000 epochs to see what kind of error rate we could achieve (and also spare my poor laptop from overheating). If you're curious about how that would work, I recommend [this video](https://www.youtube.com/watch?v=pK-LYoRwp-k) by CodeEmporium.

## Testing
Ideally, I'd like to get an aspiring actor from the Film School to participate in the sample but as a CS Major, I'm not even allowed in the building.  
Behold

<p align="center">
  <img src ="https://drive.google.com/uc?id=1REhySf37gldV79GtQPKlxkSkvN0tUmPf" />
</p>

<p align="center">
    I did my best with this one.
  <img src ="https://drive.google.com/uc?id=1CXd7avWwiK5MsSr0Yu-4TMiM5jTjMa5v" />
</p>

<p align="center">
  <img src ="https://drive.google.com/uc?id=1Y_j_6COAvrG4HohXja0EsJodmby3ec5Q" />
</p>

So majestic.  

The CNN manages to identify the easy emotions (Happy, Sad and Neutral) most of the time. There are slight variations in results when testing on others but I will need a larger sample size to draw any statistically significant conclusions.  
That being said, I have noticed that it frequently mistakes my _angry_ face for _neutral_ and _disgusted_ is a hit or a miss a lot of the time.

Despite the okay-sounding theoretical results, practical testing is not so straightforward. Especially in problems like this one with a variety of variables such as lighting, shadows, facial differences, distance from camera, camera quality, background noise and so on.

<p align="center">
  <img src ="https://imgs.xkcd.com/comics/tasks.png" />
</p>

Credit-[Randall Munroe's xkcd](https://xkcd.com/1838/)


## Future Work
I still believe there is a fair bit of work that could improve this CNN's performance. For starters, Microsoft released a [FER+ dataset](https://github.com/Microsoft/FERPlus) which is the same collection of images, only now the labels could contain more than one emotion.  
I feel this is a more accurate representation of human emotions and that it will produce better accuracy in practice.

Even beyond that, I'm curious how Apple's FaceID sensor could be potentially used for this purpose. Their face tracking hardware can capture much more information than cropping a face out of a webcam. For the time being however, I have yet to dive deep on the [ARKit API](https://developer.apple.com/documentation/arkit/creating_face_based_ar_experiences) to see what might be possible.