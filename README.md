# Knowledge Distillation for UNet

An implementation of Knowledge distillation for segmentation, to train a small (student) UNet from a larger (teacher) UNet thereby reducing the size of the network and while achieving performance similar to the heavier model.

## Results:
Dataset: [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge)

![Models trained without knowledge distillation](https://github.com/VaticanCameos99/knowledge-distillation-for-unet/blob/master/without%20knowledge%20distillation.png?raw=true)

![Models trained with knowledge distillation](https://github.com/VaticanCameos99/knowledge-distillation-for-unet/blob/master/with%20knowledge%20distillation.jpeg?raw=true)


## References
* [Distilling the Knowledge in a Neural Network -
Geoffrey Hinton, Oriol Vinyals, Jeff Dean](https://arxiv.org/abs/1503.02531)

* [Structured Knowledge Distillation for Dense Prediction -
Yifan Liu, Changyong Shun, Jingdong Wang, Chunhua Shen](https://arxiv.org/abs/1903.04197)

* [On Compressing U-net Using Knowledge Distillation -
Karttikeya Mangalam, Mathieu Salzamann](https://arxiv.org/abs/1812.00249)
