#FastAi #image
# Deep Dive into image classification
in this chapter we are going to try to classify pet breeds. we'll see what are the necessary techniques to solve this problem and future problems like this.

## The Dataset
we get the dataset from `URLs.PETS` like in [[Basics]]. just like in that lesson,
```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)
```
usually a dataset is one of 2 types:
1. the data is in individual files representing the data like images and audios
2. the data is tabular and available in one or multiple csv files.
here the data is of the first type.
```python
Path.BASE_PATH = path
path.ls()
```
```
(#3) [Path('annotations'),Path('images'),Path('models')]
```
`path.ls()` gives us what is available in the path.  here we see that there are 3 folders "Annotations", "images", and "models". we don't need the other 2 folders as the data we want is in the "images" folder.
```python
(path/"images").ls()
```
```
(#7394) [Path('images/great_pyrenees_173.jpg'),Path('images/wheaten_terrier_46.jpg'),Path('images/Ragdoll_262.jpg'),Path('images/german_shorthaired_3.jpg'),Path('images/american_bulldog_196.jpg'),Path('images/boxer_188.jpg'),Path('images/staffordshire_bull_terrier_173.jpg'),Path('images/basset_hound_71.jpg'),Path('images/staffordshire_bull_terrier_37.jpg'),Path('images/yorkshire_terrier_18.jpg')...]
```
this type of collection returned by fastai's functions belong to a class called L. they are just like lists except there is a number representing the count at the beginning and it only shows some of the dataset when printed.

we need a way to extract the pet breed from the file name. so we use regular expressions. regular expressions are used for string matching and similarity. for our files the expression `r'(.+)_\d+.jpg'` fits to extract the name.
```python
fname = (path/"images").ls()[0]
re.findall(r'(.+)_\d+.jpg$', fname.name)
```
```
['great_pyrenees']
```
`re.findall()` is a method to see if we can extract the names correctly. and as we can see we can extract the name with this method.
lets use this regular expression to label the whole dataset.
the `RegexLabeller` class in fastai is used to label based on regular expressions.
since we are going to explore image classification more deeply than before, we use the `DataBlock` class instead of `DataLoaders`. this is because it is more flexible to change.
```python
pets = DataBlock(blocks = (ImageBlock, CategoryBlock), 
				 get_items=get_image_files, 
				 splitter=RandomSplitter(seed=42), 
				 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'),'name'),
				 item_tfms=Resize(460), 
				 batch_tfms=aug_transforms(size=224, min_scale=0.75)) 
dls = pets.dataloaders(path/"images")
```
here the blocks indicate the dependent and independent variable, `get_items` has the function to get the images, splitter is how you want to split the data. we use a Random splitter. and `get_y` is how we get the dependent variable, i.e. the names of the given image pet breed. we use the `RegexLabeller` we talked about earlier.
`using_attr` takes in a function and applies it on the second argument, which is "name" here.
`item_tfms` and `batch_tfms` are used for Presizing. Presizing is used to do data augmentation on images while limiting data destruction and at the same time maintaining model performance.

## Presizing
we need our images to have the same dimensions, so that we can send it in batches to the GPU. we also want to perform minimum augmentations because if more is done, more stress on the CPU and GPU. so first we need to resize to fit in batches and then perform augmentations(minimum) on said batches. the problem is that, when augmentations are performed on resized images, many spurious empty zones are formed and degradation of data occurs.
for example when we rotate a square image 45 degrees, we have empty zones in all 4 corners which is a waste of space.
many rotating and zooming will require interpolating, to increase the clarity of the image.
to work around these issues, fastai performs presizing in 2 steps:
1. Resize images to **relatively "large" dimensions** - dimensions larger than the **target data**.
2. Compose all the augmentation operations into one and perform them at the end all together on the GPU, instead of performing all of the operations individually and interpolating multiple times.

resizing to larger dimensions mean that there won't be empty zones or degrading images after the augmentations are performed on the image. this transform works by cropping the original data into a square of size width or breadth of the original image, whichever is smaller. remember that square's size is greater than the target image after processing.
then we perform all the augmentations on the GPU as a single operation, with an interpolation in the end.
for example, 

![[Pasted image 20231220134432.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220134432.png/?raw=true)

the original image is a rectangle, we need to resize it to the same size as all images that is to be passed to the GPU
1. *crop full width or height*: this is in `item_tfms`, applied to each image before stacking them up and sending them in batches to the GPU. this is to ensure all of the images are of the same size before stacking. the crop area is chosen at random in the training set, while for the test set, it is chosen as the center part of the rectangle.
2. *Random crop and augmentations*: this is in `batch_tfms`, so it is applied to a batch that enters the GPU. here the final is image is made to fit the "target size" by random crops and augmentations. the training set does this, while the valid set, only performs the resize needed to get the target size for the model.

the third square we see in the image is the target size of the model.
in the implementation, we use the Resize function in `item_tfms` to resize all the images into 460x460 images, larger than the target size(224).
then we use `RandomResizedCrop` as a batch transform with the intended size(224). although we didn't add that explicitly, `RandomResizedCrop` will automatically be added when we include the `min_scale` parameter in the `aug_transforms` function.
we can also use pad, squish instead of crop for initial resize.

![[Pasted image 20231220135450.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220135450.png/?raw=true)

the image on the left is the target image of fastai, after both `item_tfms` and `batch_tfms`. right is done the traditional way. as we can see the, the right has empty zones and is in general a bad quality image. left one is way better.

## Debugging a Data block
its always good practice to check your data block before using it to train your model. to check we use `dls.show_batch()` to show us samples of the processed images.
```python
dls.show_batch(nrows=1, ncols=3)
```

![[Pasted image 20231220190249.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220190249.png/?raw=true)

`datablock.summary()` is also a good way to track your process and see if an error occurs. this method will attempt to create a batch with the given details
```python
pets1 = DataBlock(blocks = (ImageBlock, CategoryBlock),
				  get_items=get_image_files,
				  splitter=RandomSplitter(seed=42),
				get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'),'name')) pets1.summary(path/"images")
```
here we forgot to resize the images into the same size. so when we call `datablock.summary()` it will trace the process of making a batch and show us exactly why we got an error. here, it wouldn't be able to create a batch due to varied sizes.

once we decided that the data was good, we can immediately start with training models. its good to start with an architecture that is fast so that we can derive insights from that. one of them is the `resnet` architecture.
```python
learn = vision_learner(dls, resnet34, metrics=error_rate) learn.fine_tune(2)
```
![[Pasted image 20231220190359.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220190359.png/?raw=true)
since we use a pretrained model, we use `fine_tune()` instead of `fit_one_cycle()`
remember that the metric we gave here is `error_rate` so it gets displayed here.
now, remember from [[Neural Networks]] that this model is training by optimizing with respect to the loss function. but what is the loss function here? by default, fastai takes the cross-entropy loss as the loss function

## Cross - Entropy Loss

cross entropy is like mean squared error but better. it can handle multiple categories unlike that where it only had to predict between a 1 or a 0.
it is also faster and reliable.

### Viewing Activations and Labels
to see the activations of our model, lets first see real data from our model. that is all the numbers involved. we use `one_batch`
```python
x,y = dls.one_batch()
```
this returns the independent variables and the dependent variables of a batch.
both contain 64 elements coz, batch size is 64. so x contains 64 images all converted to 224 x 224 tensors of numbers. and y contains all the predicted values for those 64 independent variables(images).
to get the activations of our final layer(predictions) we can use `learner.get_preds()` it takes in an index(0 for training dataset and 1 for test dataset) or an iterator of batches. so we can send the batch we just got as a list to the function to get predictions of each image. `get_preds` also returns the target values but we already know that so we ignore them.
```python
preds, _ = learner.get_preds([(x,y)])
```
`preds[0]` will contain the predictions of the first image, that is 37 values, each of which are probabilities for each category. so if we add all of the predictions for one element we get 1. there are 37 elements for each prediction that is a probability for each of the 37 categories.
to transform the activations of our model into predictions like these we use the *Softmax* activation function.

### Softmax
we use this function in the final layer of our neural network to ensure that all predictions lie between 0 and 1 and that all probabilities add up to 1.
softmax is similar to sigmoid.
for a neural network that predicts if its a 3 or 7 we just passed the final activations to a sigmoid function and got values between 0 and 1. but now we need a function that can return an activation for each category.
lets try creating a simple one. activations that predict 2 things - is it a 3 and is it a 7. this is not the same as is it a 3 or not? we get activations for each class and they add up to 1.
to make it:
```python
acts = torch.randn((6,2))
acts
```
```
tensor([[ 0.6734,  0.2576],
        [ 0.4689,  0.4607],
        [-2.2457, -0.3727],
        [ 4.4164, -1.2760],
        [ 0.9233,  0.5347],
        [ 1.0698,  1.6187]])
```
the numbers on the left are activations for 3 and the right are activations for 7. we need to change it so that it adds up to 1, in such a way that bigger values have more value when transformed.
we can't use sigmoid for this because it doesn't add up to 1.
the neural networks we created were just predicting the confidence of the image being a specific number.
here we get separate activations for each category and all that matters is which is the highest and by how much?

so now how do we transform? since this problem is just the same but with 2 activations, we can just apply sigmoid to the difference between the activations, and the other column will be 1 - this.
```python
(acts[:, 0] - acts[:, 1]).sigmoid()


tensor([0.6025, 0.5021, 0.1332, 0.9966, 0.5959, 0.3661])
```
so this can be taken as the probability of it being a 3 and for the probabilities of it being a 7 is 1 - all these.
this works because, if the difference is high, so is the sigmoid and so is the probability, and if its low, the probability is also low.
now to apply it for multiple columns, we use the exponent function.
to derive the softmax function,

![[Pasted image 20231220212750.jpg]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220212750.png/?raw=true)

```python
def softmax(x):
	return exp(x)/exp(x).sum(dim=1, keepdim=True)
```
this formula means that all values add up to 1 and if the activation of a particular category is high, so is its softmax because, it increases exponentially.
Softmax is multi category equivalent of sigmoid. although there can be other functions who do the same thing of making the activations add up to 1, nothing is closely related to sigmoid like softmax. and a sigmoid curve is really smooth making it a flexible function to use. it also, as an added bonus converts the activations to a positive value as exp can't be negative. 
this function, really **picks** something out of the others, so it is definitely helpful in classification. if you want inference about images, a better option would be to add a binary activation for each category and apply sigmoid for each.

### Log Likelihood
```python
sm_acts = softmax(acts)
sm_acts

tensor([[0.6025, 0.3975],
        [0.5021, 0.4979],
        [0.1332, 0.8668],
        [0.9966, 0.0034],
        [0.5959, 0.4041],
        [0.3661, 0.6339]])
```
now that we got the predictions, we need the loss function too. in our 3s example, we simply chose the prediction opposite to the target and added it all up and took the mean and made sure it was minimum. we need to do the same here except, we just choose the probability of the right prediction and add them all up. and the result should be really high.
so here lets say for our activations,
```python
targets = tensor([0,1,0,1,1,0])

idx = range(6) 
sm_acts[idx, targ]

tensor([0.6025, 0.4979, 0.1332, 0.0034, 0.4041, 0.3661])
```
we can see that all the probabilities of the target prediction was pulled. this is because the target is just an index of one of the categories. 1 here is the index of activations where 7 is and so when 7 is the target, it can be represented as 1.
pytorch has something that does exactly the same thing as `sm_acts[idx, targ]`
that is: `nll_loss`. except it just adds a negative sign in the end.
```python
result = F.nll_loss(sm_acts, targ, reduction='none')

tensor([-0.6025, -0.4979, -0.1332, -0.0034, -0.4041, -0.3661])
```

### Taking the log
since the operations up until now contain several multiplications of negative numbers can lead to problems like [numerical underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)(whatever that means). 
so we need to transform the calculated values into larger values, to perform more calculations. so we apply log on them.
we also need to ensure difference in predictions don't get ignored, no matter how small. for example, a probability of 0.1 and 0.01 has a 10 times difference, but their actual difference is very low. so log makes sure these don't get ignored.
the main property of log is that 
log(a * b) = log(a) + log(b)
this means that any multiplication that can lead to underflow(really small numbers) or overflow(really large numbers) won't lead to that coz we apply log and turn it addition. this conversion of multiplicative to additive is very good and is used in many practical applications.

note that the log of a number tends to negative infinity as the number decreases. that means as the number decreases, log also decreases. when we apply for the loss function of ours, we need it such that, when we have the probability of the target closer to 1, the log should be less. as the probability is high, the log should be low. how to achieve this opposite relationship? answer: apply a negative sign🤷🏻‍♂
for example lets say the the probabilities are 0.1 and 0.01 for the right classes. then after applying negative log, we get 1 and 2, that is as the probabilities get further from 1(means they are getting more and more wrong, because we have the predicted probabilities of the actual classes and they need to be closer to 1), the negative log becomes larger and thus the loss also is larger.

so for the loss function, we apply negative log to the finally picked value after the `nll_loss` function. that's why its called negative log likelihood. this function assumes that you call it after applying negative log to all the probabilities and then pick it using `nll_loss`. now we'll just pick the right one and apply negative log.
```python
results = -np.log(results)
results

tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])
```
and now the NLL of all predictions are averaged and that is our loss. this is the Cross Entropy loss function
in pytorch this is available as `nn.CrossEntropyLoss`. this function does softmax then applies log and then calls `nll_loss`.
```python
loss_func = nn.CrossEntropyLoss()
```
this function takes in the original activations and the targets to give out the loss.
```python
nn.CrossEntropyLoss(reduction='none')(acts, targ)

tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])
```

we can also use this. the reduction="none" means that we are telling pytorch not to take the mean and just give us the negative log likelihood of all of them. **Remember, the mean of all the negative log likelihoods give us the loss**

another interesting feature of cross entropy is that its gradient, is proportional to the difference between the prediction and the target. that means if the difference is lesser, the change in the parameters is lesser and that means those are good parameters. also this is the same as mean squared error as its gradient is also proportional to the difference. and since the gradient is linear and not exponential, we won't see sudden jumps in the parameters, which leads to smoother training of the model.

## Model Interpretation
we can use the same confusion matrix we used in [[Image basics]]. however we have 37 categories to classify into and reading a 37 x 37 matrix is pretty hard. so we use the `most_confused` method.
```python
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=5)
```
this gives the top categories that we predict wrongly for.
```
[('american_pit_bull_terrier', 'staffordshire_bull_terrier', 10),
 ('Ragdoll', 'Birman', 8),
 ('Siamese', 'Birman', 6),
 ('Bengal', 'Egyptian_Mau', 5),
 ('american_pit_bull_terrier', 'american_bulldog', 5)]
```
the first element is the actual and the second one is the predicted. since we are not experts at breeds, we don't know if we are embarrassingly wrong or just an honest mistake. one quick google search tells us that the ones we got wrong are actually conflicting types that even pet breed experts fight on. so yea our model is pretty good. although we didn't really do anything lol

## Improving our model
there are many ways we can improve our model.

### Learning rate finder
to make sure we have the right learning rate, we use the `lr_find`. the idea was that, first we use a really low LR and we apply it on a mini batch and track the loss. then we increase the LR by a percent(usually double it) and try again and track the loss. when the loss gets worse, we know that the LR is bad. so we select a point where the loss was at its minimum
```python
learn = vision_learner(dls, resnet34, metrics=error_rate) 
lr_min,lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
```

![[Pasted image 20231220224225.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220224225.png/?raw=true)

***Our advice is to pick either:
- ***One order of magnitude less than where the minimum loss was achieved (i.e., the minimum divided by 10)
- ***The last point where the loss was clearly decreasing***

so we choose either `lr_min`/10 or `lr_steep`.
note that the learning rate in the plot is of logarithmic scale.

```python
learn = vision_learner(dls, resnet34, metrics=error_rate) learn.fine_tune(2, base_lr=3e-3)
```
now lets see how we fine tune a model's weight

### Unfreezing and Transfer learning
we saw about transfer learning in [[Basics]] . what is it actually. the thing is the learner we have is already trained on many kinds of data and has weights that help us a lot. these weights are not entirely useless for us just because, it was trained on other images. instead, these weights are used to classify features of the images and these features can be useful for us. so the only thing that is useless is the final layer. we throw it away and put in random coefficients such that we get the activations we desire. so how does the gradient descent step affects these and other coefficients? answer is it doesn't. 
pretrained models always have their previous weights frozen. they can't be adjusted using gradient descent.
by default, when we ask the model to `fine_tune`, fastai by default, freezes all the pretrained weights and only upgrades the final weights for one epoch, and then it trains like usual after unfreezing.
this approach is good, but we can do better.

first we'll use `fit_one_cycle` to train the final coefficients. in short, what `fit_one_cycle` does is to start training at a low learning rate, gradually increase it for the first section of training, and then gradually decrease it again for the last section of training
```python
learn = vision_learner(dls, resnet34, metrics=error_rate) learn.fit_one_cycle(3, 3e-3)
```
then we unfreeze the model
```python
learner.unfreeze()
```
and use `lr_find` again, because now that our model is a little trained, the previous LR will be useless.
```python
learn.lr_find()
```
now we apply the found LR.
```python
learn.fit_one_cycle(6, lr_max=1e-5)
```

now this will improve the model, but we can do even better.


### Discriminative Learning Rates
remember that the pretrained weights were already trained for a 100 epochs on millions of images. we don't need them to change that much because the features it produce might be important for us. especially the first layers. so we can use different learning rates for different layers, namely a larger one for later layers especially the random one and a smaller one for the early layers.

LR in `fine_tune` can be given as a slice object. meaning the minimum value of the slice will be used as a LR for the early layers and the max for the later one.
```python
learn = vision_learner(dls, resnet34, metrics=error_rate) learn.fit_one_cycle(3, 3e-3) 
learn.unfreeze() 
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
learn.recorder.plot_loss()
```

![[Pasted image 20231220230902.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231220230902.png/?raw=true)


this gives us wonderful results but how do we select the number of epochs?

### Selecting the number of Epochs
the first time we train a model, just select a number that you are happy to wait for. Then look at the training and validation loss plots, as shown above, and in particular your metrics, and if you see that they are still getting better even in your final epochs, then you know that you have not trained for too long.

"before the days of 1cycle training it was very common to save the model at the end of each epoch, and then select whichever model had the best accuracy out of all of the models saved in each epoch. This is known as _early stopping_. However, this is very unlikely to give you the best answer, because those epochs in the middle occur before the learning rate has had a chance to reach the small values, where it can really find the best result. Therefore, if you find that you have overfit, what you should actually do is retrain your model from scratch, and this time select a total number of epochs based on where your previous best results were found."

## Deeper Architectures
In general, a bigger model has the ability to better capture the real underlying relationships in your data, and also to capture and memorize the specific details of your individual images.
however using a large architecture might result in out of memory error. when this happens a good way to resolve it is to **reduce the batch size with the `bs` parameter in the Data loaders**
another way to efficiently train with deeper architectures is to use mixed precision training. this allows us to use less precise numbers(_half-precision floating point_, also called _fp16_)
to use this in fastai just use the `to_fp16()` after your model
```python
from fastai.callback.fp16 import * 
learn = vision_learner(dls, resnet50, metrics=error_rate).to_fp16() learn.fine_tune(6, freeze_epochs=3)
```
