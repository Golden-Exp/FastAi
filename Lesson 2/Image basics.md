#FastAi #image 
try visiting [fastai vision](https://docs.fast.ai/vision.data.html)

to search for images try `search_images_ddg`
then use `download_images` to download them
`dataloaders` are convenient but when u need to go deep and change important parameters in your model, instead of trying the model from scratch u can try the flexible DataBlock api that is to be fed to a dataloader.
![[Pasted image 20231011223304.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%202/Attachments/Pasted%20image%2020231011223304.png/?raw=true)

blocks has two features: the type of the independent variable and the type of the dependent variable
The _independent variable_ is the thing we are using to make predictions from, and the _dependent variable_ is our target.

`get_y` is the function to get the labels of the dependent variables. `parent_label` is a function of fastai that gets the name of the folder the image is in as the label

the images that we feed to the model need to be of the same size, because we are gonna stack them into a mini batch and pass them at the same time to the GPU. _Item transforms_ are pieces of code that run on each individual item, whether it be an image, category, or so forth. fastai includes many predefined transforms; we use the `Resize` transform here

A `DataLoaders` includes validation and training `DataLoader`s. `DataLoader` is a class that provides batches of a few items at a time to the GPU.`show_batch` used to show some samples
![[Pasted image 20231011224244.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%202/Attachments/Pasted%20image%2020231011224244.png/?raw=true)

All of these approaches seem somewhat wasteful, or problematic. If we squish or stretch the images they end up as unrealistic shapes, leading to a model that learns that things look different to how they actually are, which we would expect to result in lower accuracy. If we crop the images then we remove some of the features that allow us to perform recognition.
so another method is used
## Data Augmentation
Instead, what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images.
_Data augmentation_ refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes and contrast changes.

![[Pasted image 20231011225900.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%202/Attachments/Pasted%20image%2020231011225900.png/?raw=true)

```python
bears = bears.new( item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms()) dls = bears.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate) 
learn.fine_tune(4)
```
## Confusion Matrix
```python
interp = ClassificationInterpretation.from_learner(learn) 
interp.plot_confusion_matrix()
```
![[Pasted image 20231011231243.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%202/Attachments/Pasted%20image%2020231011231243.png/?raw=true)

```python
interp.plot_top_losses(5, n_rows=1)
```
this gives the top 5 images that were predicted wrong. with this we can check which images were labeled wrong since we just downloaded images after a ddg search. then we can label them rightly with the fastai GUI class
```python
cleaner = ImageClassifierCleaner(learn) 
```
![[Pasted image 20231011231847.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%202/Attachments/Pasted%20image%2020231011231847.png/?raw=true)
toggle with the buttons and do what u want. Then to save changes:

to delete (`unlink`) all images selected for deletion, we would run:
```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
```
To move images for which we've selected a different category, we would run:
```python
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```
to export and use your models elsewhere:
```python
learn.export()    #When you call export, fastai will save a file called "export.pkl"
learn_inf = load_learner(path/'export.pkl')  #importing it elsewhere
learn_inf.predict('images/grizzly.jpg')   #to predict
```
