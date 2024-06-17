#FastAi #nnbasics
refer [from-scratch-model](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch#Introduction)
this lesson focuses on creating a neural network model from scratch for the titanic dataset in Kaggle. 

# Cleaning the data

## Missing values and numerical data
first for preparing the model, we have to prepare the data for neural networks. that is all our data should be numbers. we have to multiply each column with a set of coefficients and to do this we have some basic cleaning to do.
here the missing values are filled with the modes of their respective columns.

we should also avoid tailed distributions and try to have wide spread of data
when the `fare` column of the dataset was plotted as a histogram:
![[Pasted image 20231213174723.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213174723.png/?raw=true)
this is a right tailed distribution. to even it out we apply Log to the column and 
![[Pasted image 20231213174853.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213174853.png/?raw=true)

## Categorical data
for categorical variables we use one hot encoding to create dummy variables of those columns. we do this with `pd.get_dummies()` 
so columns gender and Embarked are one-hot encoded.
finally we got our independent columns which we use to find the dependent variable survived.
![[Pasted image 20231213175540.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213175540.png/?raw=true)

# The Linear Model

once we have our columns we can turn them into tensors for us to use in the model like we did in lesson 3.

## Predictions and Loss
first to create predictions, we multiply all our rows with a coefficient for each column. here lets set a coefficient from (-0.5 to 0.5). also this time we don't need any external bias because, we have all the variables in. for example for the gender column we have both male and female separately as columns and a coefficient for each of them. nothing gets excluded. so we don't need a bias factor to balance things.
so we multiply the tensor with the coefficients only to see that the age column has very high products.
to avoid this we divide all the values of age with highest age so that all of the values of that column now comes between 0 and 1.
now,
**For a row, we calculate the predictions by multiplying each column with its coefficient and adding all the products.**
this is done for all the rows. so we get n predictions where n is the number of rows.
![[Pasted image 20231213181237.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213181237.png/?raw=true)
axis=1 means we add across columns.
```
t_indep*coeff
[[0.134, 0.213, 0.3421 ..... for all columns],
[0.238, -0.45, 02.....],
.
.
.
.
]] for all rows
then they add up using .sum(axis=1)
preds = [sum of first nested list, sum of second list... and so on]
```

the first time, these are useless coz they are random. to make use we have to do a gradient descent step. for that we need a way to calculate the loss.
here loss can be mean of all the absolute values of the predictions - the dependent variable, i.e. the correct value.
![[Pasted image 20231213182613.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213182613.png/?raw=true)

now lets declare them as functions
```python
def calc_preds(coeffs, indeps):
	return (coeffs*indeps).sum(axis=1)
def calc_loss(coeffs, indeps, deps):
	return torch.abs(calc_preds(coeffs, indeps) - deps).mean()
```

## Gradient Descent
to get the gradient descent we need the gradients of the coefficients.
we use `requires_grad_()` on the coefficients for this.
now after calculating the loss, we then call `loss.backward()` to calculate the gradients. to access the gradients we call `coeffs.grad` attribute of coefficients
```python
coeffs.requires_grad_()
loss = calc_loss(coeffs)
loss.backward()
coeff.grad -> returns the gradients
```
now with each gradient we need to adjust the coefficients by subtracting them from the gradient times the learning rate. we already discussed why we have the learning rate in [[Neural Networks]]
now first we use `with torch.no_grad()` this is to not compute gradients for the following lines of code.
```python
with torch.no_grad():
	coeffs.sub_(coeffs.grad * 0.01)
	coeffs.grad.zero_()
```
then we subtract the coefficients with their gradients times the learning rate which is 0.01 here. Fastai has a method called `lr_find()` to find a good learning rate.
any Pytorch function that ends with an underscore means that function is taking "in place".
The next time we calculate the gradients they just add up with the past gradients. so we set them to zero before the epoch ends
```python
def update_coeffs(coeffs, lr):
	coeffs.sub_(coeffs.grad * lr)
	coeffs.grad.zero_()

def one_epoch(coeffs, lr):
	loss = calc_loss(coeffs, trn_indep, trn_dep)
	loss.backward()
	with torch.no_grad():
		update_coeffs(coeffs, lr)
	print("loss: "+loss)
```

## Training the model
now before training the model we have to split the data into training validation data. this is to avoid overfitting on the training data before trying it out on the test data
```python
from fastai.data.transforms import RandomSplitter
trn_split, val_split = RandomSplitter(seed=42)(df)
trn_indep, val_indep = t_indep[trn_split], t_indep[val_split]
trn_dep, val_dep = t_dep[trn_split], t_dep[val_split]
len(trn_indep), len(val_indep)
```

now with training the model we iterate over `one_epoch()` again and again
```python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): 
	    one_epoch(coeffs, lr=lr)
    return coeffs
```

although seeing the loss slowly decreasing for each epoch does tell us that our model is doing well, we still don't have a very good metric for us the hoomans to see **exactly how good** our model is doing.
so we calculate the accuracy of our model
we'll assume that any prediction above 0.5 has survived. then we'll cross-check that with our actual survived
so, 
```python
results = val_dep.bool() == (preds > 0.5)
results.float().mean() -> accuracy
```
`val_dep` is the valid set's dependent variable. `preds` is the predictions from the valid set of the independent variable.
now, if `val_dep` is a 1(survived) it would be true in Boolean. if its a 0(died) it would result in a false.
likewise, if `preds` > 0.5(survived) it would be true. if its lesser(died), it would be false.
so this way, we are cross checking how equal the predictions and the actual values are. the mean of the number of true values(which is what is being done above by converting into float) would give us the percentage of predictions that are equal to the actual value. this is accuracy
```python
def acc(coeffs): 
return (val_dep.bool()==(calc_preds(coeffs,val_indep)>0.5)).float().mean()
```
==ignore the indentation error above==

now lets take a look at our predictions
![[Pasted image 20231213213207.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213213207.png/?raw=true)
we can see that some of them are less than 0 and greater than 1. this is not convenient for us as the predictions should fall under the range 0 to 1.
this can be achieved with the sigmoid function.
![[Pasted image 20231213213327.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%205/Attachments/Pasted%20image%2020231213213327.png/?raw=true)
whatever the x value, the value returned will always be between 0 and 1. this is perfect for our predictions. so we pass all our predictions through the sigmoid function
```python
def calc_preds(coeffs, indeps):
	return torch.sigmoid((coeffs*indeps).sum(axis=1))
```
so all the rows' predictions are between 0 and 1 now and the other process continues as usual.
that's it for the linear model.

# Implementing the Neural Network

## Matrix Product
instead of multiplying each column with it's respective coefficient and adding them all, we can do a matrix multiplication which is the same as doing that.
```python
def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)
```
we use the @ symbol for matrix multiplication. this is much faster than the regular way because Pytorch optimizes the function.
that means to matrix multiply, the coefficients shouldn't be a 1-D tensor, but a 2-D vector with 1 column. this can be done like this.
```python
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()
```
same for the dependent variable tensor
```python
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]
```
the independent variables are already vectors

## The Neural Network
first lets create a neural network with one hidden layer.
there should be coefficients for each layer. the first set of coefficients should be equal to the number of columns. then that should give out n outputs. where n can be any number we want. a higher number means more flexibility but more time.

there can be more outputs for the first layer this time because we are doing matrix multiplication. remember, 
lets say we have a matrix 
A  -> (100 x 12)
B -> (12 x 1)
the first number is the number of rows and the second is the number of columns.
this setup is the same as the one we used for the linear model above. (100 x 12) means there are 100 rows and 12 columns of independent variables.
and the number of coefficients should be the same as the number of columns so (12 x 1)
Now, matrix multiplication can only be done when
**the number of columns in A is equal to the number of rows in B**
here 12 == 12 so it works.
if this condition is satisfied then 
**the resulting product of the matrix multiplication will contain the same number of rows as A and same number of columns as B**

A(x, y) @ B(y, z) the condition is true
so the resulting product C will have x rows and z columns(C(x, z))

by keeping the number of columns as 1 in the coefficient matrix, we make sure we get just one output for one multiplication,  which is, the prediction for that row.

but for our simple neural network that we are trying, we need the first layer to spit out multiple outputs which go in as inputs for the second layer that takes in that many inputs and spits out one output that passes through the sigmoid function and come out as a prediction

so when creating the coefficients we create a matrix of x, y which means x rows and y columns. x being the number of columns in the independent variable matrix and y being the number of outputs you want to spit out.
the resulting "outputs" for the next layer is a matrix of (n, y) n being the number of rows

then the second layer should be a matrix of (y, 1) because it should be matrix multiplied with the outputs of the first layer(n, y) and the resultant matrix will be (n,1) that is one prediction for each row.

so the end result is the same as the linear model which is (n,1). but the process is expanded so that there is more room for flexibility and corrections

this is the basic premise of a neural network

```python
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return 
    layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()
```

the first layer gives out `n_hidden` outputs. the second layer gives out 1 output. here the second layer is also added with a constant for even more flexibility. these are all the coefficients
also, the reason why the first layer is divided by the number of outputs is when we sum up all the values in the next layer, it will be similar to its original value. this is normalization

now for the predictions:
```python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
```
res is the output matrix of the first layer with `n_hidden` columns and the same number of rows as `indeps`.
then we pass that matrix through a `relu` for introducing the non-linearity that helps to adapt to any function and the resulting matrix is `res`.
now `res` is matrix multiplied with layer 2 with `n_hidden` rows and 1 column.
the output is one prediction that is added to a constant term.
this final output is passed to a sigmoid function to get the predictions between 0 and 1
and the gradient descent should be
```python
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

## Deep Learning
our previous model only had 1 hidden layer. that really isn't that 'deep' to be considered deep learning.
so we add multiple layers using the same matrix multiplication technique as before
```python
def init_coeffs():
    hiddens = [10, 10]  # <- set to the size of each hidden layer. the length of this list is how many hidden layers are there
    sizes = [n_coeff] + hiddens + [1] #here sizes = [ncoeffs, 10, 10, 1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)]
    for l in layers+consts: l.requires_grad_()
    return layers,consts
```
now here, layers is a list containing each matrix of each layer. those are
1. the matrix for `n_coeff` inputs and 10 outputs
2. the matrix for 10 inputs and 10 outputs
3. the matrix for 10 inputs and 1 output
also for each layer a constant is made to add up with.
now for the predictions
```python
import torch.nn.functional as F

def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i!=n-1: res = F.relu(res)
    return torch.sigmoid(res)
```

enumerate(layers) returns the index(i) and the matrix(l)
first the independent variables are matrix multiplied with the first layer and added to the constant.
then if it is not the last layer, the `relu` is added. this is because practically, there is no need for a non-linearity in the last layer, where we actually predict instead of increasing the non linearity. this is done by the past layers and not the last layer.

then we pass through sigmoid for 0 to 1
and the gradient descent is done like
```python
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

and that is how a neural network is constructed.
***The biggest differences in practical models to what we have above are:
- ***How initialization and normalization is done to ensure the model trains correctly every time
- ***Regularization (to avoid over-fitting)
- ***Modifying the neural net itself to take advantage of knowledge of the problem domain
- ***Doing gradient descent steps on smaller batches, rather than the whole dataset. 

