#FastAi  #nnbasics 
## Basic classification without a neural network
use `untar_data(dataset)` to get the data from fastai
`path.ls()` -> shows the content of the path
`Image.open(path)` -> this is a method of the python Image class used to open images. printing this would show us the image
when we convert an image to an array or a tensor it becomes a 2-D array with numbers representing the color of each pixel
after converting the image into an array/tensor we can make it into a DataFrame and then we can color code it to show what number represents what color
```python
im3_t = tensor(im3) 
df = pd.DataFrame(im3_t[4:15,4:22]) 
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```
![[Pasted image 20231101164934.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101164934.png/?raw=true)

to classify whether an image is a three or a seven we can first see what is the average value is for each pixel in a 3 and 7. then we can compare the given image to the 'ideal' 3 and 7 and whichever is similar will be the image.
to display the array into an image we use fastai's `show_image()` function
`show_image(array of pixels)`
```python
seven_tensors = [tensor(Image.open(o)) for o in sevens] 
three_tensors = [tensor(Image.open(o)) for o in threes]
```
using list comprehension we open each image and convert it into a tensor and store it. so the `seven_tensor` is a tensor with each element being the tensor version of the image
this is essentially a 2-d list as in a group of tensors in a list.
```python
stacked_sevens = torch.stack(seven_tensors).float()/255 
stacked_threes = torch.stack(three_tensors).float()/255
```
now to compute the mean of each pixel position we first convert the list into a rank 3 tensor.
its rank 3 because there are 3 attributes: the length and breadth of the image and the no. of images.
the rank is the no. of dimensions and the shape is the size of each axis(dimension)
to do this we use the stack method. and then we covert the pixel numbers to float and have it be between 0 and 1 so we divide by 255
now we compute the mean of all pixel positions
```python
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)
```
![[Pasted image 20231101170638.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101170638.png/?raw=true)
![[Pasted image 20231101170539.png]]  ![[Pasted image 20231101170927.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101170539.png/?raw=true)

this the 'ideal' three and seven
now to see if an image is a three or seven we can calculate the differences between the pixel position and add them up. if the difference sum is heigh its further from ideal three and so is not a three. however sum differences might be negative and when we add them it might cancel and result in a zero. so to avoid this we can
- add the absolute value of the differences 
- add the squares of the differences
```python
dist_3_abs = (a_3 - mean3).abs().mean() 
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
```

to check the loss, Pytorch provides a function called F inside `torch.nn.functional`.
```python
F.l1_loss(a, b) or F.mse_loss(a, b).sqrt()
```
here l1 means abs loss and mse is mean squared error loss. we are adding a square root so that we can compare both errors.
#### Broadcasting in tensors
when two tensors of different ranks are made to do a simple operation, the tensor with the lower rank will expand itself into the higher rank. it doesn't actually allocate memory to do so but thinks it is.
```python
def mnist_distance(a,b): 
	return (a-b).abs().mean((-1,-2)) 
valid_3_dist = mnist_distance(valid_3_tens, mean3)
def is_3(x): 
	return mnist_distance(x,mean3) < mnist_distance(x,mean7)
accuracy_3s = is_3(valid_3_tens).float() .mean() 
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()
```
here a is the tensor and b is the ideal 3. due to broadcasting, the ideal 3 is subtracted and the mean of the absolute values of all the image is returned.
so we get a tensor containing the distance of each image from the ideal 3 very quickly thanks to broadcasting
then we calculate the accuracy. to do so we calculate on average how many trues we get and that is the accuracy. we pass the tensor of images to is_3 and it returns a tensor of trues and false.
this is converted to float. true to float is 1 and false is 0.
then the mean is taken which is the no. of 1's by the total no. which is the accuracy.
the accuracy of this simple model is shown to be 95%
## Neural Networks

the previous method doesn't learn and can't improve. with deep learning we go to a different approach. here we a assign a parameter/weight for each position and we can calculate what the weight actually needs to be so that we get a good value if the pixel is a three's or not, if its not a three. to improve we reiterate
![[Pasted image 20231101173123.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101173123.png/?raw=true)

![[Pasted image 20231101173148.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101173148.png/?raw=true)

### Stochastic Gradient Descent (SGD)
it is a method of neural networks to calculate the loss and upgrade the parameters and update the loss.
imagine a loss function and the weight to be a random point on the function. to calculate the "better" weight so that the loss function is at its lowest we can update our weight little by little by calculating the slope.
![[Pasted image 20231101173614.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101173614.png/?raw=true)

the slope or the gradient tells us what will happen to the loss if we increase or decrease the parameter. then with that we can update the parameter to decrease the loss.
the gradient is the slope of the function. it tells us an idea of how far we can change the weights to decrease the loss. **it isn't exact.**
so we use a learning rate to update our weights
`w -= gradient(w) * lr`
the subtraction allows us to steer the weight into the direction of the lowest point of the loss function.
![[Pasted image 20231101174218.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101174218.png/?raw=true)
if the learning rate's too high we might avoid the direction completely and go away from our required point
![[Pasted image 20231101174256.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101174256.png/?raw=true)
so we need to be careful when setting the learning rate.
![[Pasted image 20231101174443.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101174443.png/?raw=true)
the `require_grad()` means the gradient should be calculated
and `.backward()` calculates the gradient.
to get the gradient use the attribute, `.grad`.
```python
def init_params(size, std=1.0): 
	return (torch.randn(size)*std).requires_grad_()
weights = init_params((28*28,1))
bias = init_params(1)
def linear1(xb): 
	return xb@weights + bias   #the @ is used for matrix multiplication
preds = linear1(train_x)  #here broadcasting is used to get the preds of all images
corrects = (preds>0.0).float() == train_y #assumed that if greater than 0 its 3
corrects.float().mean().item() #calculating accuracy
```
now we need to determine the loss function. although accuracy can be loss function it shouldn't this is because accuracy doesn't change much when u change the weight. accuracy only changes when the predictions change. so a little change in the weight wont tell us whether we improved.
so we need a loss function that decreases when we change our weights accordingly.
a simple loss function for that would be
```python
def mnist_loss(predictions, targets): 
	return torch.where(targets==1, 1-predictions, predictions).mean()
```
this function is written in such a way that when the target is 1, it returns 1 - the prediction, so the more close we are to 1 the lesser it returns, likewise when the target is 0, it returns the prediction, so if we predict closer to 0, it returns lesser.
thus this function returns a lesser value, the closer the prediction is to a target.
the mean of all the results after it is done on the predictions is the loss.
here the function assumes that the predictions are between 0 and 1. but the matrix multiplication returns random numbers. to make the predictions come between 0 and 1 we use the sigmoid function
![[Pasted image 20231101193034.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101193034.png/?raw=true)
this converts any number to a number between 0 and 1
```python
def sigmoid(x): 
	return 1/(1+torch.exp(-x))
```
it always returns between 0 and 1
```python
def mnist_loss(predictions, targets): 
	predictions = predictions.sigmoid() 
	return torch.where(targets==1, 1-predictions, predictions).mean()
```
now how many data shall we calculate the loss and optimize simultaneously? the answer comes in the form of mini-batches. We determine the batch size early and send in that size of data for optimization. more b.s = longer time
a Dataloader can convert any collection into a batches of tensors. this is used for optimization
typically a dataset is a collection of tuples containing the dependent variable and the independent variable.
we pass the dataset to the dataloaders to split it into batches.
```python
def calc_grad(xb, yb, model): 
	preds = model(xb)  #predictions
	loss = mnist_loss(preds, yb)  #passed through sigmoid and loss calculated
	loss.backward() # gradients calculated
def train_epoch(model, lr, params): 
	for xb,yb in dl:     #reiterating over each batch in the dataloader
		calc_grad(xb, yb, model) 
		for p in params: #updating params
			p.data -= p.grad*lr 
			p.grad.zero_()   #when we reiterate the grads add up whenever we calculate it. so we change it to zero each time. in pytorch the '_' symbol means in-place
def batch_accuracy(xb, yb): 
	preds = xb.sigmoid()    
	correct = (preds>0.5) == yb 
	return correct.float().mean()
def validate_epoch(model): 
	accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl] #returns accuracy
	return round(torch.stack(accs).mean().item(), 4)
for i in range(20): 
	train_epoch(linear1, lr, params) 
	print(validate_epoch(linear1), end=' ') #reiterating over and over again.and printing accuracy each time.
```

this can be done much simply using classes and objects. fastai provides a class called SGD that does it.
```python
linear_model = nn.Linear(28*28,1) 
opt = SGD(linear_model.parameters(), lr)    #creates the object
train_model(linear_model, 20)    #updates the parameters of the object
```
fastai also provides Learner.fit() which trains the model. we need a learner for that.
to create a learner we pass in the dataloaders, the model, optimization function, loss function, and any metrics
```python
learn = Learner(dls, nn.Linear(28*28,1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
learn.fit(10, lr=lr)
```
this will train the model for us and give the accuracy
## Non linearity
what we have discussed so far is a simple linear classifier. to change it so that it can adapt to more complex functions we add a non linearity
```python
def simple_net(xb): 
	res = xb@w1 + b1    
	res = res.max(tensor(0.0)) 
	res = res@w2 + b2 
	return res
```
it is simply a max function between two matrix multiplications
the max function returns the same number if its positive or a zero. this the non linearity we added
it is also known as a *rectified linear unit*

![[Pasted image 20231101203117.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101203117.png/?raw=true)

```python
w1 = init_params((28*28,30)) # returns a tensor of 30 tensors each having 784 random numbers. all of the 30 tensors are matrix multipled to return 30 outputs 
b1 = init_params(30)  #the 30 outputs are added to 30 biases
w2 = init_params((30,1))  #gives 30 random variables
b2 = init_params(1) #gives one bias
```
here we define 30  28 * 28 weights and multiply them with the data. so we get 30 activation outputs. now we do RELU and pass it to the next matrix multiplication and since there are 30 that means, there should be 30 input activations for w2.
note that due to broadcasting when passing the entire dataset, all of these happen to each image.
the idea here is that using many linear layers we have our model do more computation and thus adapt to more complex functions. however if we stack up multiple linear layers again and again it would be of no use because we can replace the whole network with a single layer with different parameters because it is a linear layer.
so we add a non-linearity that is the RELU. with this we create separate layers each which aren't replaceable. it is also said that any complex function can be replaced with a number of RELUS because they can add many non linearities such that it is approximately the complex functions.
```python
simple_net = nn.Sequential( nn.Linear(28*28,30), nn.ReLU(), nn.Linear(30,1))
```
![[Pasted image 20231101204045.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%203/Attachments/Pasted%20image%2020231101204045.png/?raw=true)
