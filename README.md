# Image-Morpher
#### By: Arav Shah

### Examples

Some sample output gifs are available in the "static/output_images" folder.

The coresponding input images can be found in the "static/ex_imgs" folder.

### Usage

To use the image morpher, clone the github directory and run “website.py” in your terminal. Open a web browser and paste the link from the terminal into a new tab. From there, upload two square pictures (with file type: “.jpg,” “.jpeg,” or “.png”) and select the length of the gif you desire. Your resulting gif and input images will be displayed under a new url after about 3 minutes. You may then continue to use the image morpher, and it will continue displaying all past results until the server is restarted.

#### Notes: 
The image morpher will resize all input images to 224 by 224 pixels. Therefore, you must upload a square photo to avoid the image being distorted and the model being unable to make accurate key point predictions. As a result, your final gif will also be 224 by 224 pixels.

When selecting two images of faces, the faces should be the primary focus of the images, and the faces on both images should be roughly the same size for best results.

You may run into issues running the web interface if you do not have the required libraries installed in your environment. These libraries include Flask, PyTorch, and Dask.

## Technical Details

#### Image Morphing:
In order to compute a gif smoothly transitioning from one image to another, this program creates 45 individual frames (computed in parallel using Dask to improve run time). Each frame is a blend of image1 * w and image2 * (1-w), where w is some weight between 0 and 1. 

The process of creating each individual frame requires some linear algebra. Without going into too much detail, once each image is labeled with corresponding key points from the model, we compute a Delaunay triangulation for each set of points. This Delaunay triangulation ensures that each point is connected to two other points forming a triangle, and it ensures that the triangulation will connect all corresponding points for the two images in the same way. With this triangulation, we are able to morph the shape of a triangle on image1 to the average shape of that triangle and the corresponding triangle on image2 using an affine transformation. We then iterate through every pixel in both triangles to fill in the pixel values adjusted by weight, w, for image1 and weight, (1-w), for image2. Doing this for every triangle results in one morphed image.

The image morpher creates the frames by using 45 evenly distributed weight values between 0 and 1. Once all output frames have been computed, they are simply saved as a gif and displayed back to the user on the website.

#### Model:
The model is the other critical part of this image morpher. In order to compute the Delaunay triangulation required, we need a set of corresponding key points (i.e. a point on face1 referencing the bottom of person1’s left earlobe should have a corresponding point on face2 referencing the bottom of person2’s left earlobe. These corresponding key points need to encompass the entire face and need to be accurate. 

These points are predicted by a neural network trained on the “ibug_300W” face dataset. This dataset consists of over 6000 faces labeled with corresponding key points. After data clean-up and augmentation, these images were used to train PyTorch’s resnet18 neural network for 20 epochs. After rigorous testing and adjusting hyperparameters, the final model had a training loss of  0.000219, and a validation loss of 0.000519 (after key points were converted to floats between 0 - 1 rather than 0 - 224). This model was then saved and downloaded and is being used for the key point prediction of this project (it can be found in the “models” folder).

Note: All code for the backend of this project was written in Python.

#### Website: 
The website itself was created primarily in Flask and html. As mentioned in the code comments, some inspiration for this code and design was taken from Prof. Donald Patterson's Flask tutorial available on Youtube. Note: As mentioned above, the web interface is available but has not been deployed, so it must be run locally. 
