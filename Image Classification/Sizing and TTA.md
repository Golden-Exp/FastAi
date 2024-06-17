#FastAi #image 

we are going to see more ways to improve our image classification model, especially when we are building from scratch or using a pretrained model to predict entirely new things.
# The data
we use the `imagenette` dataset.
```python
from fastai.vision.all import * 
path = untar_data(URLs.IMAGENETTE)

dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
				   get_items=get_image_files, get_y=parent_label,
				   item_tfms=Resize(460),
				   batch_tfms=aug_transforms(size=224, min_scale=0.75)) dls = dblock.dataloaders(path, bs=64)
```
this dataset contains totally random images. now lets use a model that is from scratch and has the same number of final activations as our number of categories.
```python
model = xresnet50(n_out=dls.c) 
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
				metrics=accuracy) 
learn.fit_one_cycle(5, 3e-3)
```
When working with models that are being trained from scratch, or fine-tuned to a very different dataset than the one used for the pretraining, there are some additional techniques that are really important. 

# Normalization
when training a model, it helps when the data we use is normalized. that is, it has a mean of 1 and a standard deviation of 0. but most images have values between 0 and 1 or between 0 and 255. 
lets grab some data and average them to see how they are. we'll
```python
x,y = dls.one_batch() 
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])
```
```
(TensorImage([0.4842, 0.4711, 0.4511], device='cuda:5'),
 TensorImage([0.2873, 0.2893, 0.3110], device='cuda:5'))
```
yea the means are not close to 0 and the standard deviations are not close to 1.
its fairly easy to normalize.