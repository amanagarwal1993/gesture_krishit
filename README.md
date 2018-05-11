# gesture_recg

The following library was used to access the Azure cloud and CustomVision services.
* azure.cognitiveservices.vision.customvision.training

The training and prediction keys along with the API call URL are hidden from the .py file. Please contact if needed.

Basic CV and Machine Learnng libraries are imported, as visible in the code.

Large parts of the code are self-explanatory. Various parameters and variables are initialised with required information in comments.

The removeBG function does majority of the pre-processing. Information about the functions can be found in the OpenCV documenatation.

The dimensions of the ROI and training images is 320px*384px.

An alternate classification output is available using exisiting OpenCV functions such as Convex Hull and Convexity Defects.

The main result is classification by the CNN into 4 classes: One, Two, Five, Okay. More classes can be added with ease once training data is available.

To capture initial background, press B.

To get classification result, press c.

To reset static background, press R.

to close windows, press Esc.
