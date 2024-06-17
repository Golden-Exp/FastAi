#FastAi #tabular

what is collaborative filtering?

when we have a number of users and a number of products and to choose which products are more likely to be used by which users and recommend them, we use collaborative filtering. this is what usually happens when recommending movies in OTT(Netflix), recommending products to buy(Amazon), etc.

In collaborative filtering, we look at the products that the user currently used and liked, find other users that had similar tastes and recommend the products that those users liked.

***"For example, on Netflix you may have watched lots of movies that are science fiction, full of action, and were made in the 1970s. Netflix may not know these particular properties of the films you have watched, but it will be able to see that other people that have watched the same movies that you watched also tended to watch other movies that are science fiction, full of action, and were made in the 1970s. In other words, to use this approach we don't necessarily need to know anything about the movies, except who like to watch them."***

## The data
for this lesson the data is from [Movie lens](https://grouplens.org/datasets/movielens/)
the columns are user, movie, rating and timestamp
![[Pasted image 20231214115145.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214115145.png/?raw=true)

here is the cross tabulation of the data
![[Pasted image 20231214115218.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214115218.png/?raw=true)

the empty spaces are what we are gonna fill in with our model and see whether a particular user would like that particular movie.

one way to calculate is if we know how much each user likes each category a movie falls into. like for example lets say the three important categories a movie falls under are *action*, *comedy*, and *murder mystery*. (these are just examples)
then we could map each movie and user's taste accordingly. each category would have a value from -1 to 1 where -1 is hating and 1 being enjoyed and 0 being neutral.
then we can represent a movie, for example, *Aliens in the Attic* as 
```python
Aliens_in_the_Attic = np.array([0.7, 0.7, -0.8])
```
where 7 in action and comedy means the movie is actually kind of action and comedy and a -8 means its **not** a murder mystery.
now lets represent a user that likes action comedy movies and hates murder mysteries as 
```python
user = np.array([0.9, 0.8, -0.9])
```
then the rating of ***Aliens in the Attic*** from this user can be predicted as 
```python
rating = (user*Aliens_in_the_Attic).sum()
```
that rating will be 1.92, which is kind of high. if we set the categories and coefficients accordingly we can make it so that we get a value between 0 and 5, 0 being the lowest and 5 being the highest.
however this is a good method because, when the users likes are lower for a certain category, the rating also gets lowered, which is what we need.
the multiplying of the columns and adding them up is known as *dot product* and is the basis of matrix multiplication.

these *categories* are known as **Latent factors**

## Latent Factors
calculating the latent factors is the same as the neural network we built in [[From Scratch Model]]
the steps involved for a Collaborative filtering are:
1. randomly initialize the latent factors and the number of them. how many to set is a question we'll discuss later. in this example we use 5. each user will have 1 values for each "category" or "latent factor". and each movie will also have 1 value for each latent factor. so we need 5 times the number of movies of random numbers for the movies and 5 times the number of users for the users.![[Pasted image 20231214121424.png]]
the above picture shows how the coefficients are set.
2. To predict the rating we use dot product as discussed above. so to predict the score that user 14 will give for movie 27, we need to perform dot product of the latent factors of user 14 and movie 27 and the result is the rating. this is how all the ratings are calculated. with this method, if the score for action of a movie is high and the user's action score is also high the resulting rating will also be high. however is the same movie is used for another user with low action score, the rating will also be low.
3. next we calculate the loss. we can use any metric for this. here we use Mean squared error.
that's all that is required for us to perform stochastic Gradient Descent.
first the predictions are calculated using dot product. next the loss is calculated using any metric and the derivatives of the parameters based on the loss is calculated. then the parameters are optimized with this derivative and the process repeats until we get a lower loss.

## Creating the Data loaders
after merging the movie id table and the names table we get a Data frame called ratings
![[Pasted image 20231214122356.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214122356.png/?raw=true)
with this let's build a data loaders for collaborative filtering
```python
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
```
by default it takes in the first s=column as the users, the second as the item and the third as the ratings. here we change the items to be the names of the movie rather than the id.
a batch of this data Loaders will be like
![[Pasted image 20231214122600.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214122600.png/?raw=true)

we can't use a cross tab representation for deep learning although its good for us to understand.

we use simple matrices for both latent factors
```python
n_users = len(dls.classes["user"])
n_movies = len(dls.classes["title"])
n_factors = 5

movie_factors = torch.randn(n_movies, n_factors)
user_factors = torch.randn(n_users, n_factors)
```

to look up the latent factors we just need the index of the user and the index of the movie. however our models can't look up indices. they just matrix multiply and calculate losses and move on.
another way to find the required latent factors is to matrix multiply with a one hot encoded vector of the required index.
```python
index_3 = one_hot(3, n_users).float()
user_factors@index_3 -> this will return the user factors at index 3
```
`one_hot(x, n)` will return an array of n integers where n-1 elements are zeros and the element at index x will be a 1.
when we multiply this with the user factors matrix's transpose, we get the latent factors of the user factors matrix at index x.

this is because `user_factors` transpose is (5, `n_users`). the matrix `index_3` is (`n_users`, 1) with its index 3 element being 1.
when matrix multiplication takes place, all the elements in row 1(which are all the values of latent factor 1 because this is a transpose) are multiplied with respective elements in the one hot encoded matrix which is zero everywhere except at index 3. so all elements become 0 except index 3 which becomes the value of the first element in the resultant product matrix. this continues for all 5 rows.
so when matrix multiplying we get a matrix of (5,1) which is just an array with 5 elements which are the latent factors at index 3.
so now we know that we perform matrix multiplication with a one hot encoded vector to get the latent factors of a particular movie/user.
now to define some terms
##### Embedding matrix
this is the matrix that contains a unique row of latent vectors for each user and movie. there is a separate embedding matrix for users and a separate one for movies
the Pytorch library assigns an index for each movie and user and uses that while creating the one hot encoded matrix to get the latent factors. 
![[Pasted image 20231214135227.png]] ![[Pasted image 20231214135249.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214135227.png/?raw=true)
##### Embeddings
this is the result of matrix multiplying with the one hot encoding to get the latent factors of a particular movie or user.
the columns `user idx` and `movie idx` is created by the library and are used to create the one hot encoding.
to get the embeddings for user 14 we multiply its one hot encoded vector of index 1 with the user's embedding matrix. this results in just the latent factors for user 14
(we looked up the latent factors of index 1 in the array(user embeddings) using matrix multiplication)
same for the movies. then to predict ratings we just perform dot product between user embedding and movie embedding
![[Pasted image 20231214135331.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214135331.png/?raw=true)

***"Embedding actually means look something up in an array. So, there's a lot of things that we use as deep learning practitioners, to try make you as intimidated as possible, so that you don't wander into our territory and start winning our Kaggle competitions***
***Unfortunately, once you discover the simplicity of it, you might start to think that you can do it yourself and it turns out, you can" :)***
-Jeremy Howard

## Collaborative Filtering from scratch
let's create a class for performing dot product between a user and a movie.
```python
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
```
Module is a Pytorch module we need to inherit for our class to use the method Embedding and forward.
forward is called upon whenever we call the class. this will be useful.
here, we initialize that `user_factors` is an embedding matrix of `n_users` rows and `n_factors` columns. which is 5 in here.
then `movie_factors` is an embedding matrix of `n_movies` rows and `n_factors` columns, which is 5.
note that the input for this model is a tensor x, where the first column
`x[:, 0]`, are the users and the 2nd column `x[:, 1]` are the movies. and there are the same number of rows as each batch size.
what we need to do is multiply the user factors of the given users with the movie factors of the given movie(which is what is being defined in forward)
the "looking up" of which movie or which user is done with the embedding matrix when we say `user_factors(x[:, 0])` -> all the latent factors of the first column(users) are "looked up" and loaded in users.
same for movies.
then we multiply them to get a matrix with the same number of rows as the passed batch size and 5 columns. then we add them across the columns to the get the rating of each user for that movie.

now lets create a learner for this model. here we use 50 latent factors instead of 5.
```python
model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func = MSELossFlat())

#fitting our model
learn.fit_one_cycle(5, 5e-3)
```

##### The Forward function
when we fit our model, pytorch calls upon the forward function we defined and passes in each batch to that function and gets back the predictions for that batch.

now time for some changes in the model.
firstly we can force the predictions to be between 0 and 5. it was found that 0 to 5.5 would be a better range. this is because a sigmoid can never hit 1. although it gives us values between 0 and 1. we are gonna use a sigmoid to restrict the values between 0 and 5 and even when we use 5.5 as the end limit it can't give us a 5.5 so, we can focus on the maximum output being 5.

when we multiply the sigmoid function by a number we get a function that spits numbers between 0 and that number. boom.
the `sigmoid_range` function returns the input passed through the sigmoid times the total range.
```python
#documentation of sigmoid_range function
def sigmoid_range(x, low, high): 
	"Sigmoid function with range `(low, high)`" 
	return torch.sigmoid(x) * (high - low) + low
```

```python
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)
```

another thing we could change is that we could add biases to each user and each movie. this makes adds a unique constant value for each user and movie, which is also really good, because some users might always have positive feedbacks. some might always have a negative feedback. likewise a movie can be, in general, a bad movie.
these individual factors can be represented by biases
so now, 
```python
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)
```
`user_bias` is a constant term for each user. same with `movie_bias`.
now the resultant predictions are added with their respective movies and respective user biases. for this we use an embedding matrix for biases also.
then as usual we pass it through the `sigmoid_range` function.
however training this model results in failure.
![[Pasted image 20231214160028.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214160028.png/?raw=true)

as you can see the valid loss increases after a point. this indicates overfitting. to avoid this we use another regularization technique like data augmentation. that is weight decay.
### Weight Decay
this technique involves the addition of the sum of all the weights squared to the loss function. this will make it so that the optimization makes sure that the weights to become too high because we want our loss to be much low as possible.
how would this prevent overfitting?
this is because we are penalizing large weights. large weights mean that when that particular column is somewhat high, the weight makes sure that the prediction is high. this kind-of leads to over fitting. lets say a particular user factor's weights are too high. then, every time we need to predict the ratings given by that user, there is a high chance that it would be good rating due to that high weight. this leads to overfitting.
this doesn't corelate to whether the user gives a good rating for every movie. coz that is being managed by the user bias.
![[Pasted image 20231214161610.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214161610.png/?raw=true)

here we can see that as a becomes larger, the plot becomes sharper and sharper. now consider a to be the weight. as a becomes larger, a ** 2 becomes sharper and can overfit with a very complex function with its sharpness.
so now to add this to the loss
```python
loss_with_wd = loss + wd*((parameters**2).sum())
```
squaring all the weights and adding them up would be too tiring on the CPU. so, instead we can directly add 2 * parameters * wd to the gradient of each parameter.
since we are doing derivative on parameters.
```python
parameters.grad += wd*2*parameters 
```
here wd is a hyperparameter weight decay. if its high the weights must be much lower and if the weights are too restricted, it won't be a good model. however if wd is too low, the weights can expand more resulting in overfitting. so we need a mix of both and set a good wd.
also we don't need 2 * wd as we are directly providing the wd, it can just be some constant. this can be passed while we fit the model
![[Pasted image 20231214163112.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214163112.png/?raw=true)
now it doesn't overfit.

### Our own embedding Module
so far we just used the embedding function of pytorch for creating an embedding matrix.
lets create it on our own.
when defining a pytorch module to define a parameter we use the `nn.Parameter()` function to let the module know its a weight.
so now lets create a function for initializing parameters in our embedding module
```python
def create_params(size):
	return nn.Parameter(torch.zeros(*size).normal(0, 0.01))
```
now the Dot Product class becomes
```python
class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        movies = self.movie_factors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)
```

## Interpretations from the embeddings and bias
```python
movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes['title'][i] for i in idxs]
```
this shows us the movies with the lowest bias scores. meaning movies that were generally bad. this also means even if a user seemingly liked movies that fall under the genres of this movie, the user may not like this movie
output:
```
['Children of the Corn: The Gathering (1996)',
 'Home Alone 3 (1997)',
 'Crow: City of Angels, The (1996)',
 'Mortal Kombat: Annihilation (1997)',
 'Cable Guy, The (1996)']
```

likewise we can do the same for good movies
```
['Titanic (1997)',
 "Schindler's List (1993)",
 'Shawshank Redemption, The (1994)',
 'Star Wars (1977)',
 'L.A. Confidential (1997)']
 ```

now there is something called PCA(Principal Component Analysis) which is used to pull out some underlying directions from the embedding matrix. the details were not given, however if you are interested click [here](https://github.com/fastai/numerical-linear-algebra)
the interpretations from doing PCA are given by
```python
g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['title'].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0,fac1,fac2 = movie_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()
```
![[Pasted image 20231214165812.png]]from the above diagram we can infer a few things. all the critically acclaimed movies like *the godfather*, *Shawshank Redemption* group near the right side of the plot. and all the action driven movies like *terminator*, *Star Wars* group near the bottom. All the movies that are dialogue driven are grouped above and mainstream movies like *mission impossible* and *Independence day* are grouped to the left.(https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214165812.png/?raw=true)
now we didn't program it to classify based on this, but our model did this by taking into account what users liked what movies.
obviously it didn't know that all those movies belonged to that genre and grouped them accordingly but instead it grouped movies that were similar.
which is honestly, really cool imo

***No matter how many models I train, I never stop getting moved and surprised by how these randomly initialized bunches of numbers, trained with such simple mechanics, manage to discover things about my data all by themselves. It almost seems like cheating, that I can create code that does useful things without ever actually telling it how to do those things!***

we can do all this using Fastai's collab learner
## FastAi.collab
```python
learner = collab_learner(dls, n_factor=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)
learn.model -> gives out the names of layers
```
![[Pasted image 20231214170919.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231214170919.png/?raw=true)
we can do the same interpretations as above like this
```python
movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]
```
output:
```['Titanic (1997)',
 "Schindler's List (1993)",
 'Shawshank Redemption, The (1994)',
 'Casablanca (1942)',
 'Silence of the Lambs, The (1991)']
```

### Embedding distance
another interesting thing about embeddings is that we can see which movie is the most similar to another movie.
movie similarity can be defined by the similarity of users that like those movies. And that directly means that the distance between two movies' embedding vectors can define that similarity.
```python
movie_factors = learn.model.i_weight.weight
idx = dls.classes['title'].o2i['Silence of the Lambs, The (1991)']
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['title'][idx]
```
this gives us the most similar movie to Silence of the Lambs, which is 
*8 Seconds (1994)*

## Bootstrapping
Bootstrapping of a collaborative model is a common problem, where we don't know past data about a user.
up until now, the user factors are calculated based on the movies they watched and rated. however when a new user with no history comes in, what do we do?
there are many answers to this but all of them circles back to the same saying:
*use your common sense*
you can try averaging all the latent factors and give it to the new user, however this might not always work out. coz the average of lets say, sci-fi latent factor might be high, but that particular latent factor may not be common for everyone.

another way might be to set a person as average and use their latent factors each time a new user comes in.

but the most common way is to ask questions at the beginning, to determine the latent factors. after asking the questions, we can pass the answers to a model where the dependent variables are the latent factors and the independent variables are the answers to the questions
that's why sites like Netflix and Amazon asks survey questions when you sign up for the first time.

## Deep learning and Collaborative Filtering
to turn this into a deep learning model, we first need to concatenate the embeddings "lookup" and the data given before passing them to the neural network.
this time the embeddings for users and movies can be of different sizes.
Fastai's `get_emb_sz` function is used to the get the recommended sizes of each embedding matrix

```python
embs = get_emb_sz(dls)
class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)
model = CollabNN(*embs)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)
```
recall from [[Neural Networks]] that `nn.Sequential` is used for neural networks.
in the forward method, we calculate the embeddings for the batch, concatenate them and pass it to the layers. then we pass them through sigmoid for the predictions.

`collab_learner` has a parameter `use_nn`, which can be used to create neural networks directly
```python
learn = collab_learner(dls, use_nn=True,
					   y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)
```
here there are 2 layers one with size 100 and another with size 50.

## Categorical Embeddings
the embeddings that we used for the collaborative filtering model are known as entity embeddings
these entity embeddings can be used to replace one hot encoding of categorical variables when we pass it to a neural network. that is because, these embeddings make the categories "continuous".
using them in any model is simple. All we have to do is first do a one hot encoding of the categorical columns and replace the 0s and 1s with continuous variables. for example, when one hot encoding is done on a categorical column(using `pd.get_dummies` or something) we get columns corresponding to each category. now replace each 1s with a random number from 0 to 1. same for 0 except, if its another category use another random number to represent that specific category. i know this sounds confusing.
![[Pasted image 20231215102426.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231215102426.png/?raw=true)
one hot encoding has been done on "cat2" column. then if you see "cat2_2" which corresponds to the column representing category 2 of the 19 categories in cat2.
now a random number is assigned for each of the 19 categories in the "cat2_2" column. same is done for all the encoded columns. this is entity embedding.
now we just concatenate the embedded columns with the original dataset and pass it to the model
![[Pasted image 20231215102704.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231215102704.png/?raw=true)
as you can see the one hot encoded column "cat2_1" representing category 1 in cat2 column has 19 unique values representing the 19 categories. same with the one hot encoded column "cat2_2".
![[Pasted image 20231215103744.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231215103744.png/?raw=true)
now why do all this?
one hot encodings represent the categories in a categorical form. but entity embeddings represent categories in a continuous form which is much better for neural networks.
and because of this we can map similar categories together using the concept of "embedding distance".
![[Pasted image 20231215103815.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%207/Attachments/Pasted%20image%2020231215103815.png/?raw=true)
all these embeddings are automatically done by Fastai's models to categorical variables.
