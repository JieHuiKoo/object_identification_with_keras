# object_identification_with_keras
About: This package takes in an image containing a single object, runs through the machine learning model, and outputs the class and confidence

To run, execute the following command
```
rosrun object_identification_with_keras object_identify.py
```
The node will publish the following topics:
1) armCamera/nearestObject_Class - The class of the object within the image
2) armCamera/nearestObject_Confidence - The confidence (out of 100) of the classification
   <br />
   <br />
