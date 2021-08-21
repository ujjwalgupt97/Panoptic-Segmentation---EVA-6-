# Capstone Project - Panoptic Segmentation

## Part 1:

## Questions:

- We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention **(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**
- We also send dxN Box embeddings to the Multi-Head Attention
- We do something here to generate NxMxH/32xW/32 maps. **(WHAT DO WE DO HERE?)**
- Then we concatenate these maps with Res5 Block **(WHERE IS THIS COMING FROM?)**
- Then we perform the above steps **(EXPLAIN THESE STEPS)**
- And then we are finally left with the panoptic segmentation

![image](https://user-images.githubusercontent.com/51078583/130270831-87ada104-f5d6-4280-82c1-d359603e7295.png)

## Solution:

In the above image the represented portion is the feed to the Decoder of the Tranformer and how bounding boxes created will be used for generating panoptic segmentation scales. 

### Q1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention **(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**

The H and W is nothing but the Height and width of the encode image passed. Encoding passed into the decoder is as a result from the backbone which is a Resnet50 model flattened with the Fully Connected layers being removed. The H/32 and W/32 is a result of the Resnet model been sent . Suppose the image is sent is passed through he Resnet block scaling the image to H/32, W/32, 2048. 

Let the original image be of the dimensions 3xHxW where 3 is nothing but the RGB colour dimensions of the image. Before the image goes through the encoder-decoder model it is passed through the backbone as mentioned before where the ResNet50 model generates a lower resolution activation map of size H/32, W/32, 2048. The resultant output when passed to the encoder is reduced to a lesser number of channel from 2048 to 256. Since the encoder expects 1D input the H/32 and W/32 is merged into one and passed into the Encoder. 

The Encoder takes in the modified feed passed through the Resnet block and compressed 1D image and process the positional and patch embeddings. The encoder works on a multi head attention model where in DeTr we have 8 heads. The Result of the encoder generated is passed to the decoder. Along with these inputs of D, H/32, W/32 we have the object queries as additional inputs of dimensions dxN to the Multi-Head Attention

### Q2. We do something here to generate NxMxH/32xW/32 maps. **(WHAT DO WE DO HERE?)**

As discussed earlier there are two set of inputs to the Decoder :- the encoded image which is passed from the encoder and then comes the object queries or the box embeddings. For training our image we will be sending both the stuff i.e. the background like the sky, grass etc and the things i.e. is the objects like the steel bars, bricks etc. 

The box embeddings is used to locate these bounding boxes based on the attention and store the data in the set . Using the bipartate matching the set of actual values and predicted values the decoder trains the model to find the correct/ exact location of the bounding boxes. The maximum number of bounding boxes that can be made is the number of box embeddings provided i.e. 100 in the case of DeTr. 

Here in the dimension **NxMxH/32xW/32**:
- N is the number of these Box embeddings generated
- M is D/N where D is the channels passed from the encoded image
- H is nothing but the Height of the original image
- W is nothing but the width of the original image 

Now once these encoded images are passed through and trained to generate a bounding box for the stuff and things inside the image, we need to pass and create head maps of each item located with the bounding box. Note that since we are considering both the stuff and things in the image we will find bounding boxes covering throughout the image. 

Let’s take the example from the image given in the description . 

![image](https://user-images.githubusercontent.com/51078583/130319329-58c63117-91a0-47f2-aafd-a2a5ec9c8ad4.png)

As in the image above we can see there are 4 bounding boxes created using the object embedding and the encoder-decoder model . Now we pass the bounding boxes to the head maps which we get from the Backbone structure and train to generate segmentation scales on the bounding box for the image individually and combine them later

### Q3. Then we concatenate these maps with Res5 Block **(WHERE IS THIS COMING FROM?)**

As discussed in the previous answer we work on the attention maps. There are the activation map  generated from the backbone that is the Resnet 50  and is stored. The activation heat maps are extracted from the layer 2,3,4 and 5 of the Resnet block and the encoded images or the attention heat maps gets sampled from NxMxH/32xW/32 to NxH/4xW/4 by passing it continuously through these Resnet layer and eventually getting the desired segmented scale image. Each number of bounding box is passed based on that class and activation maps help train to generate the segmentation scale for that particular class based on the bounding box and activation maps.

Here is the representation in the description diagram:

![image](https://user-images.githubusercontent.com/51078583/130320293-f1fce556-7663-4066-ba6e-69a081f0595c.png)

We have 4 bounding box and hence the 4 generated attention maps each concentrating to a particular class. Then the Resnet block which stores the activation maps of these class is used to train over the bounding box to generate the NxH/4xW/4 masks logits for each of the bounding box which is the segmentation scale of that particular bounding box.  


### Q4 Then we perform the above steps **(EXPLAIN THESE STEPS)** And then we are finally left with the panoptic segmentation

![image](https://user-images.githubusercontent.com/51078583/130270831-87ada104-f5d6-4280-82c1-d359603e7295.png)

The above image is nothing but the overall explanation of how the panoptic segmentation will be applied using the DETR approach. As discussed in the above discussions , this is merely the summation of all. 

Summarising, we have the Resnet50 backbone which helps us generate the 1D sequential input for the image serving as the input for the encoder. The (H/32XW/32)xD is distributed into multiple heads and attention is applied to this 1D sequential input. The encoded image is then passed into the decoder where is concatenated with the object queries/ object embeddings. The decoder follows the same attention pattern as that of the encoder dividing into 8 heads and applying attention on the combined input as a result help building the bounding boxes throughout the image . Remember we are sending in both the stuff and the things as a result the entire image will be covered with the bounding boxes. Since the dimensions of the object embedding is 100, we can have a maximum of 100 bounding boxes in the image . 

The number of bounding boxes created is equal to the attention map. Each attention focuses on the class pertaining to that particular class within the bbox. These attention maps are sampled to generate the segmentation scale for that particular class in the bbox. In the end we have segmentation scales on these individual attention map which is later concatenated to generate the final panoptic segmented image .


## My Approach. 

Here are the steps what I intend to follow to solve the Capstone project. These might change as I proceed with the Part 2 and Part 3 of the Project . 

- Gather the Dataset. The dataset contains both the stuff and the things . Things from the images we annotate and stuff from DETR. 
- Convert the Dataset in COCO format . 
- Train the dataset passing it through a modelwhich trains to predict bounding boxes for both these stuff and things . 
- Prepare the Resnet block . 
- Freeze the weights of the bounding box prediction. 
- Pass in the bounidng boxes image through a model to create the attention maps over the Resnet to generate the Segmentation scale. 
- Concatinate these maps to generate the final Panoptic segmented images. 

P.S. - These steps may change as I go ahead with the implementation . This is my initial intended approach for the Capstone Project . 

# Contributor:

Ujjwal Gupta (ujjwalgupta97@gmail.com)

