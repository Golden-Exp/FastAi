#FastAi #image
in this chapter we are going to look at other problems while image classification like classifying into multiple categories and regression in images, where our predictions will be continuous variables instead of categorical.
# Multi-Label Classification
these type of problems have images where we have to identify the object in the image. there might be multiple objects or no objects in the image at all. lets see how things change from regular classification to these types of problems

## The Data
the dataset here is the PASCAL dataset, which can have more than 1 object in its images.
```python
from fastai.vision.all import * 
path = untar_data(URLs.PASCAL_2007)
```
here the names are in a csv file
```python
df = pd.read_csv(path/'train.csv') 
```
now how do we convert from Data frame to a Data Loaders? we use a Data Block like we used in [[Image Classification(Pet Breeds)]]. we have 2 main classes for representing training and valid sets

1. `Dataset`: collection that returns a tuple of independent and dependent variable
2. `DataLoader`: an iterator that provides stream of mini batches, where each mini batch is a tuple of independent and dependent variable.

and on top of these we have:
1. `Datasets`: An object that contains a training `Dataset` and a validation `Dataset`
2. - `DataLoaders`:: An object that contains a training `DataLoader` and a validation `DataLoader`

since `Dataloader` is basically Datasets in batches, we'll do Datasets first and go to `Dataloaders`

lets build a `DataBlock` first from the very beginning
```python
dblock = DataBlock()
```
we can create a `Datasets` from this
```python
dsets = dblock.datasets(df)
len(dsets.train), len(dsets.valid)
```
when you print an element of the train attribute, we see that it is a tuple containing 2 elements. it should be the independent variables and the dependent variable, but instead now we have two rows of the data as two elements. this is because, by default it assumes that we have the independent variable then the dependent variable in the data frame. to change this we have to use the `get_x` and `get_y` functions which tell the block how to get the variables.
```python
dblock = DataBlock(get_x = lambda r: r['fname'],
				   get_y = lambda r: r['labels']) 
dsets = dblock.datasets(df) 
dsets.train[0]
```
we are telling that the `fname` is the independent variable and the `labels` is the dependent variable we are predicting. now when we print the element we get a tuple with the image name and the category we are predicting.
```
('005620.jpg', 'aeroplane')
```

but this still isn't quite right. because the dependent variable should be the path of the image file and not just the name of the file. so,
```python
def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ') 
dblock = DataBlock(get_x = get_x,
				   get_y = get_y) 
dsets = dblock.datasets(df) 
dsets.train[0]
```
```
(Path('/home/jhoward/.fastai/data/pascal_2007/train/002844.jpg'), ['train'])
```
now to actually  open the image and convert them into tensors we use the blocks parameter.
```python
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
				   get_x = get_x, get_y = get_y) 
dsets = dblock.datasets(df) 
dsets.train[0]
```
```
(PILImage mode=RGB size=500x375,
 TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
```
nice, so we got the images as the independent variable as `PILImage` and the label as independent `TensorMultiCategory`.
the dependent variable is an array of all categories, with the values being 1 when the image belongs to that category. this is *one hot encoding*. this is different from when use the `CategoryBlock`. when using `CategoryBlock` it returns the index of the category in the vocab.
here we get a list with as many elements as the categories and 1 if the image belongs to that category.
the reason why we use a list of all the encodings instead of just a list of the indices that are in the image is that there will be varied list sizes and pytorch requires same size overall.
when we pass the index of the available 1 to the vocab, we get the category
```python
idxs = torch.where(dsets.train[0][1]==1.)[0] 
dsets.train.vocab[idxs]
```
```
(#1) ['dog']
```
also there is a column named `is_valid` for rows to be in the validation dataset. we didn't use it so lets do that. we define a function for this and give it to splitter
```python
def splitter(df): 
	train = df.index[~df['is_valid']].tolist() 
	valid = df.index[df['is_valid']].tolist() 
	return train,valid 
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
				   splitter=splitter, get_x=get_x, get_y=get_y) 
dsets = dblock.datasets(df) 
dsets.train[0]
```
splitter functions should always return a list on idices.
also now we have to resize all of the images to the same size. so, we need to use the `item_tfms` parameter.
```python
dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
				   splitter=splitter, get_x=get_x, 
				   get_y=get_y, 
				   item_tfms = RandomResizedCrop(128, min_scale=0.35)) 
dls = dblock.dataloaders(df)
```
now we have our `dataloaders`. remember that `dataloaders` has tuples of each batch, where the first element of each tuple is the tensor of 64 images and the next element is a collection of the one hot encodings of the 64 images in the batch.

now we'll see the new loss function for this problem.

## Binary Cross Entropy
a learner object takes in 4 things: the model, a `DataLoaders` object, an `Optimizer`, and the loss function to use. we have a `dataloaders` and we know how to optimize using SGD. and the model we use will be `resnet` as usual. so we'll focus on the loss function. to see that lets see the activations of a vision learner.
```python
learn = vision_learner(dls, resnet18)
```
to see the activations we can just pass a batch of independent variables to our model in learner.
```python
x,y = dls.one_batch()
acts = learn.model(x)
acts.shape
```
it tells us that the shape is [64, 20]. that is true because a batch contains 64 elements so 64 predictions and each prediction means an activation(probability) foreach category. so there are 20 activations for each image signifying the 20 categories.
```python
activs[0]
```
```
TensorBase([-1.4608,  0.9895,  0.5279, -1.0224, -1.4174, -0.1778, -0.4821, -0.2561,  0.6638,  0.1715,  2.3625,  4.2209,  1.0515,  4.5342,  0.5485,  1.0585, -0.7959,  2.2770, -1.9935,  1.9646],
       grad_fn=<AliasBackward0>)
```
these activations are still not yet made to be between 0 and 1 but we know how to do that. we pass it to a sigmoid.
we also know how to transform them in such a way that they all add up to 1(softmax). but they are not useful here because as we saw in [[Image Classification(Pet Breeds)]] softmax really picks 1 from all which is not what we are doing. instead we use `Binary_Cross_Entropy`
```python
def binary_cross_entropy(inputs, targets): 
	inputs = inputs.sigmoid() 
	return -torch.where(targets==1, inputs, 1-inputs).log().mean()
```
this is the same as the loss function we used in [[Neural Networks]] with the negative log taken. this can be applied to multiple targets thanks to broadcasting.
so what happens is it compares the sigmoid(activations) to the targets, which is also a one hot encoded list.
so it picks the inputs corresponding to the 1s and the 1 - inputs corresponding to 0. then it takes the log and applies negative. then finally adds 'em up and takes mean.
by default, fastai takes this as our loss function.
since this is a multilabel problem, we can't use accuracy as our metric. this is because accuracy assumes that there is only 1 answer for 1 image.
instead lets use this:
```python
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()
```
for now the threshold is 0.5. meaning, if its greater that 0.5, the predictions become a 1 and if its lesser they become 0. then we check if we got the one hot encoding correct, and more the number of equal encodings, more the mean and more the accuracy.
if we want to pass this as our metric, but want to change the threshold, we use the partial keyword. this returns a function object with changed default values.
`func = partial(func, a=2)`
```python
learn = vision_learner(dls, resnet50, metrics=partial(accuracy_multi,
													  thresh=0.2)) learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
```
now, the threshold is really important because, a low threshold means many values are mistook and taken as 1s and high threshold means that the model must pedict very confidently for the prediction to be taken as a 1.
to see how many thresholds perform for each model, we can use the validate method to see how it does.
```python
learn.metrics = partial(accuracy_multi, thresh=0.1) 
learn.validate()

(#2) [0.10477833449840546,0.9314740300178528]

learn.metrics = partial(accuracy_multi, thresh=0.99) 
learn.validate()

(#2) [0.10477833449840546,0.9429482221603394]



preds,targs = learn.get_preds()
xs = torch.linspace(0.05,0.95,29) 
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs] 
plt.plot(xs,accs);
```
![[Pasted image 20231221150224.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231221150224.png/?raw=true)
we see that as the threshold increases, the accuracy increases till 0.5. then it slowly decreases. that concludes multi labels. now onto regression

# Regression

models that concern images doesn't only mean image classification. we can do many things with given data. image regression means the dependent variables are floating point numbers and do not belong to a set of categories. there can be many problems like predicting captions for a given image or predicting images for a given caption
to improvise and do anything, we have to have a good understanding with the Data Block API. now lets move on to regression with images	
we are going to predict the key point of an image, given the image of a person.
a key point refers to the center of the face of the person in the image.

## The Data
```python
path = untar_data(URLs.BIWI_HEAD_POSE)
path.ls().sorted()

(#50) [Path('01'),Path('01.obj'),Path('02'),Path('02.obj'),Path('03'),Path('03.obj'),Path('04'),Path('04.obj'),Path('05'),Path('05.obj')...]
```
all are different folders and each folder has the same person with different poses.
```python
(path/'01').ls().sorted()

(#1000) [Path('01/depth.cal'),Path('01/frame_00003_pose.txt'),Path('01/frame_00003_rgb.jpg'),Path('01/frame_00004_pose.txt'),Path('01/frame_00004_rgb.jpg'),Path('01/frame_00005_pose.txt'),Path('01/frame_00005_rgb.jpg'),Path('01/frame_00006_pose.txt'),Path('01/frame_00006_rgb.jpg'),Path('01/frame_00007_pose.txt')...]
```
here if we go to a folder, we see that there are different frames, each of them come with an image and a text file containing the pose. lets write a function to get the pose from an image
```python
img_files = get_image_files(path) 
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt') img2pose(img_files[0])
```
now lets see an image
```python
im = PILImage.create(img_files[0]) 
im.shape
im.to_thumb(160)
```
![[Pasted image 20231221151924.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231221151924.png/?raw=true)
we need another function to extract the center from the pose.
```python
cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6) 
def get_ctr(f): 
	ctr = np.genfromtxt(img2pose(f), skip_header=3) 
	c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2] 
	c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2] 
	return tensor([c1,c2])
get_ctr(img_files[0])
```
we can pass this function as `get_y` to the data block, coz it gets the dependent variable we need.
note that the validation set shouldn't be random. since we need our model to generally select the center regardless of person, we'll use only one person's whole images as the validation dataset.

The only other difference from the previous data block examples is that the second block is a `PointBlock`. This is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates as it does to the images:
```python
biwi = DataBlock( blocks=(ImageBlock, PointBlock),
				 get_items=get_image_files, get_y=get_ctr,
				 splitter=FuncSplitter(lambda o: o.parent.name=='13'),
				 batch_tfms=aug_transforms(size=(240,320)), )

dls = biwi.dataloaders(path) 
dls.show_batch(max_n=9, figsize=(8,6))
```
![[Pasted image 20231221152806.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231221152806.png/?raw=true)

and that's all for the data

## Training the model
now lets train like usual
```python
learn = vision_learner(dls, resnet18, y_range=(-1,1))
```
the `y_range` parameter is set here, meaning we restrict the values from -1 to 1. we'' see how `y_range` works in [[Collaborative Filtering]].

now lets see the loss function
```python
learn.loss_func
```
it gives us `MSELoss`. which is a good loss function for this. because, it gives the loss based on how close we are to the actual values and that is how we need it to be. we can also change the loss function using the `loss_func` parameter
here we can also use the loss function as our metric because it is pretty intuitive.
```python
learn.lr_find()
```
![[Pasted image 20231221153637.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231221153637.png/?raw=true)

```python
lr = 1e-2 
learn.fine_tune(3, lr)
```
we see that the final validation loss is 0.000036, which is really low, so we did a good job.
to verify the results use the `show_results`
```python
learn.show_results(ds_idx=1, nrows=3, figsize=(6,8))
```
![[Pasted image 20231221153812.png]](https://github.com/Golden-Exp/FastAi/blob/main/Image%20Classification/Attachments/Pasted%20image%2020231221153812.png/?raw=true)

***"It's particularly striking that we've been able to use transfer learning so effectively even between totally different tasks; our pretrained model was trained to do image classification, and we fine-tuned for image regression."***

in conclusion,
make sure you think hard when you have to decide on your choice of loss function, and remember that you most probably want:
- `nn.CrossEntropyLoss` for single-label classification
- `nn.BCEWithLogitsLoss` for multi-label classification
- `nn.MSELoss` for regression