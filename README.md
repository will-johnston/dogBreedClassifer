# dogBreedClassifer
An iOS application that uses a converted Keras model to identify the breed of dogs.

This application uses a converted Keras model, which can classify between 120 breeds with an accuracy of 90%.

## To run

Download the xcode project. Then plug in your iPhone, go to Product -> Destination -> your iPhone, 
and then build and run the application. Make sure the <model>.mlmodel is in your workspace.

Note: If you would like to retrain the model using different configurations, run trian_model.py. Then convert the model to a coreml model by running coreml_model_converter.py

Here is the app in action!


![img_0072](https://user-images.githubusercontent.com/22545572/44748905-fb470f80-aade-11e8-9ae4-e7361f5d08b1.jpg)

