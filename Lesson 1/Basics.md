#FastAi
![[Pasted image 20231011185932.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%201/Attachments/Pasted%20image%2020231011185932.png/?raw=true)
*basic model*

![[Pasted image 20231011190115.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%201/Attachments/Pasted%20image%2020231011190115.png/?raw=true)
*more formal chart*

![[Pasted image 20231011215405.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%201/Attachments/Pasted%20image%2020231011215405.png/?raw=true)
## from fastai.vision.all import *
imports necessary info for vision models

```python
path = untar_data(URLs.PETS)/'images'
```

>path is the variable that stores the path
>untar_data is a fastai function that "unboxes" the dataset and stores it in the path
>URLs is a class of fastai with datasets. PETS is one of them
>'images' is the name of the folder

now we feed the data to make it into a dataloader which we can feed to a learner to start training
```python
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))
```

there are different dataloaders. this is for vision. Imagedataloaders should always be used with a function like .from_name_func to get the labels

>path is default path for the learner to use
>next is the names
>valid pct is the split of train and valid
>label_func is the function of labels

now comes Transforms. this contains code that needs to be applied during training. two types of transforms
			>item_tfms applied to each item
			>batch_tfms applied to batches at a time

the batch size, bs, is equal to how many batches we are sending to the model
all the elements in a batch is "seen" by the model parallelly

```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
```

tells fastai to create a _convolutional neural network_ (CNN) and specifies what _architecture_ to use (i.e. what kind of model to create), what data we want to train it on, and what _metric_ to use

metric measures the quality of the model and is different from loss. Loss is used to modify parameters while metrics are for the humans to understand how the model is functioning

**vision_learner** also has a parameter `pretrained`which sets the weights in the model to values that have already been trained by experts to classify images.
A model that has weights that have already been trained on some other dataset is called a _pretrained model_. **You should nearly always use a pretrained model**, because it means that your model is already very capable. 

When using a pretrained model, `vision_learner` will remove the last layer, since that is always specifically customized to the original training task and replace it with one or more new layers with randomized weights, of an appropriate size for the dataset you are working with. This last part of the model is known as the _head_.

Using a pretrained model for a task different to what it was originally trained for is known as **_transfer learning_.**

```python
learn.fine_tune(1)
```

here fine_tune is used because the model is pretrained so we are not fitting but fine tuning it.
1. Use one epoch to fit just those parts of the model necessary to get the new random head to work correctly with your dataset.
2. Use the number of epochs requested when calling the method to fit the entire model, updating the weights of the later layers (especially the head) faster than the earlier layers (which, as we'll see, generally don't require many changes from the pretrained weights).

The _head_ of a model is the part that is newly added to be specific to the new dataset. An _epoch_ is one complete pass through the dataset.
then it displays
the results after each epoch are printed, showing the epoch number, the training and validation set losses and any _metrics_ 
the accuracy of the model is always checked on the validation dataset

![[Pasted image 20231011215309.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%201/Attachments/Pasted%20image%2020231011215309.png/?raw=true)

in case of tabular data mostly only **fit_one_cycle** is used because there is no pretrained model for that specific data.

should be careful when choosing valid and test data. 
examples:
	>when dealing with time series use future dates for valid 
	>when dealing with images, try using totally new characters/people in the valid test coz the model might overfit and just predict smth whenever it sees the same character.
	