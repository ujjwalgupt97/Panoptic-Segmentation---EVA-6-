# Capstone Project - Panoptic Segmentation

## Part 1:

## Questions:

- We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention **(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**
- We also send dxN Box embeddings to the Multi-Head Attention
- We do something here to generate NxMxH/32xW/32 maps. **(WHAT DO WE DO HERE?)**
- Then we concatenate these maps with Res5 Block **(WHERE IS THIS COMING FROM?)**
- Then we perform the above steps (EXPLAIN THESE STEPS)
- And then we are finally left with the panoptic segmentation

![image](https://user-images.githubusercontent.com/51078583/130270831-87ada104-f5d6-4280-82c1-d359603e7295.png)

## Solution:

In the above image the represented portion is the feed to the Decoder of the Tranformer and how bounding boxes created will be used for generating panoptic segmentation scales. 


### Q1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention **(FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**

The H and W is nothing but the Height and width of the encode image passed. Encoding passed into the decoder is as a result from the backbone which is a Resnet50 model flattened with the Fully Connected layers being removed. The H/32 and W/32 is a result of the Resnet model been sent . Suppose the image is sent is passed throught he resnet block scalling the image to H/32, W/32, 2048. 

Let the original image be of the dimensions 3xHxW where 3 is nothing but the RGB color dimentions of the image. Before the image goes throught the encoder-decoder model it is passed through the backbone as mentioned before where the ResNet50 model generates a lower resolution activation map of size H/32, W/32, 2048. The resultant output when passed to the encoder is reduced to a lesser number of channel from 2048 to 256. Since the encoder expects 1D imput the H/32 and W/32 is merged into one and passed into the Encoder. 

The Encoder takes in the modified feed passed through the Resnet block and compressed 1D image and process the postional and patch embeddings. The encoder works on a multi head attention model where in DeTr we have 8 heads. The Result of the encoder generated is passed to the decoder. Along with these imputs of D, H/32, W/32 we have the object queries as additional inputs of dimensions dxN to the Multi-Head Attention

