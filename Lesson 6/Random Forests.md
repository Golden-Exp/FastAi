#FastAi #tabular 
this lesson focuses on random forests, creating one and looking at its important features. we use the titanic dataset and the bulldozers competition in Kaggle

# Titanic
lets first look at what decision trees are with the help of the titanic dataset.

## Data Preprocessing
Unlike in [[From Scratch Model]] we don't have to create dummy variables for categorical columns here. we just have to convert the columns to categorical using `pd.Categorical()`
the other changes are the same as before.
![[Pasted image 20231215175628.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215175628.png/?raw=true)

## Binary Splits
the basis of a random forest is a decision Tree and the basis of the decision Tree is the binary split.
A binary split is when we split the data based on a threshold such that rows having values above the threshold are predicted to be one of the dependent variables while below means the other.
For example, lets split on the gender and build a very simple model saying that all women survived and men didn't 
from the survived data we can see that most survivors were female.
![[Pasted image 20231215180026.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215180026.png/?raw=true)

first the training and validation split.
```python
from numpy import random
from sklearn.model_selection import train_test_split

random.seed(42)
trn_df,val_df = train_test_split(df, test_size=0.25)
trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)
```
we change the categories to codes here.(cat is the list of categorical columns)

```python
def xs_y(df):
    xs = df[cats+conts].copy()
    return xs,df[dep] if dep in df else None

trn_xs,trn_y = xs_y(trn_df)
val_xs,val_y = xs_y(val_df)
```
now that the split has been done, lets predict
```python
preds = val_xs.Sex == 0
from sklearn.metrics import mean_absolute_error
mean_absolute_error(val_y, preds)
```
this gives us an error of 0.215. which is not bad for a single split. lets try splitting somewhere else, like the fare column
![[Pasted image 20231215180359.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215180359.png/?raw=true)
looking at the box plot we can see that survived people, tend to have a median fare of 3.2 and dead people had a median of 2.5. lets split at 2.7 where if its above 2.7, it means survival.
```python
preds = valid_xs.Logfare > 2.7
mean_absolute_error(val_y, preds)
```
this gives an error of 0.3363 which is greater than the last split. now, 
lets create a function to see which is the best column to split on
```python
def side_score(side, y):
	tot = side.sum()
	if tot <= 1: return 0
	return y[side].std()*tot
def score(col, y, split):
	lhs = col<=split
	return (side_score(col[lhs]) + side_score(col[~lhs]))/len(y)
```
we calculate the score of one group of the split using the `side_score` function and calculate the overall score of the split by adding both the groups' score.
side score returns the standard deviation times the number of rows. this is to see if the rows in the groups are similar or not. if they are similar, then its good and the standard deviation will be lesser because there is less spread.
then we multiply with the total sum to get a larger "score".
then we add the scores of both sides and divide by the length of the data frame to normalize.
so now we can score the splits of all the columns.
however we still don't know on what do we split the columns. we can have the computer decide it for us by going through all the unique values in a column and seeing which value to split on so that the score is the lowest.
we can implement it like this
```python
def min_col(df, nm):
	col, y = df[nm], df[dep]
	unq = col.dropna().unique()
	scores = np.array([score(col, y, o) for o in unq if not np.isnan(o)])
	idx = scores.argmin()
	return unq[idx], scores[idx]

cols = cat+cont
{o:min_col(df, o) for o in cols}
```
the output splits and their scores are:
```
{'Sex': (0, 0.40787530982063946),
 'Embarked': (0, 0.47883342573147836),
 'Age': (6.0, 0.478316717508991),
 'SibSp': (4, 0.4783740258817434),
 'Parch': (0, 0.4805296527841601),
 'LogFare': (2.4390808375825834, 0.4620823937736597),
 'Pclass': (2, 0.46048261885806596)}
```
we can see that sex has the best score to split on.
this is the OneR approach. to split on the best column.

## Creating a Decision Tree
how do we improve on the previous binary split? we split more.
lets further split the already split two groups based on the scores of the columns other than sex.
```python
cols.remove("Sex")
ismale = trn_df.Sex==1
males,females = trn_df[ismale],trn_df[~ismale]
{o:min_col(males, o) for o in cols}
{o:min_col(females, o) for o in cols}
```
```
{'Embarked': (0, 0.3875581870410906),
 'Age': (6.0, 0.3739828371010595),
 'SibSp': (4, 0.3875864227586273),
 'Parch': (0, 0.3874704821461959),
 'LogFare': (2.803360380906535, 0.3804856231758151),
 'Pclass': (1, 0.38155442004360934)}

{'Embarked': (0, 0.4295252982857327),
 'Age': (50.0, 0.4225927658431649),
 'SibSp': (4, 0.42319212059713535),
 'Parch': (3, 0.4193314500446158),
 'LogFare': (4.256321678298823, 0.41350598332911376),
 'Pclass': (2, 0.3335388911567601)}
```
from this we can see that the next best split for the female group is to split the `Pclass` column on 2(class 2) and for males it is on `Age` on 6.
this is how we create a decision Tree. Rather than doing this manually we can do this using sklearn
```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz

m = DecisionTreeClassifier(max_leaf_nodes=4).fit(trn_xs, trn_y);
```
`max_leaf_nodes` means we perform two levels of splits. 
we can also draw the Decision tree.
```python
import graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=2, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))

draw_tree(m, trn_xs, size=10)
```
![[Pasted image 20231215183108.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215183108.png/?raw=true)

just like how we did manually, the first split is on sex then for males its age and females its Pclass.
1. the first line in each internal node is the condition of the split.
2. the second line is kind of like the score condition we made.
3. the third line is the number of rows that come under this group 
4. and the 4th line is how many of them survived and how many died. 
[survived, perished].
also blue color means they most definitely survived while orange meant they didn't.
more the opacity more sure we are of the result.
the `gini` function can be defined as
```python
def gini(cond):
    act = df.loc[cond, dep]
    return 1 - act.mean()**2 - (1-act).mean()**2
```
***What this calculates is the probability that, if you pick two rows from a group, you'll get the same `Survived` result each time. If the group is all the same, the probability is `1.0`, and `0.0` if they're all different:***

lets try more nodes
![[Pasted image 20231215184430.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215184430.png/?raw=true)

these splits are the reason why we kept the categories. because if we needed to split on rows where embarked is "C" we just need to set the condition embarked<=0 because the code for "C" is 0. however if we need to split on "Q" we first need to split on C and not C. then split the not C on Q and not Q. because "Q " is encoded with 1.
for this reason it might be better to use dummy variables for cases like this
***"In practice, I often use dummy variables for <4 levels, and numeric codes for >=4 levels."***

## The Random Forest
we can't make decision trees more bigger than 50 nodes. so what do we do?
we use *Bagging*
*Bagging is a way of averaging the predictions of a bunch of models*
when we average a lot of predictions of totally unrelated models, we get the value of the "true" prediction. that is because the average of totally random errors of all the models will tend to be 0
to make unrelated models we use different subsets of data for each of them.
```python
def get_tree(prop=0.75):
    n = len(trn_y)
    idxs = random.choice(n, int(n*prop))
    return DecisionTreeClassifier(min_samples_leaf=5).fit(trn_xs.iloc[idxs], trn_y.iloc[idxs])

trees = [get_tree() for t in range(100)]
```
each element the list trees is a Decision Tree.
to get the predictions we average all the predictions of all trees on the same validation dataset.
```python
all_probs = [t.predict(val_xs) for t in trees]
avg_probs = np.stack(all_probs).mean(0)
mean_absolute_error(val_y, avg_probs)
```

sklearn's Random forest does the same except it also selects a random subset of the columns to split on.
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100, min_samples_leaf=5)
rf.fit(trn_xs, trn_y);
mean_absolute_error(val_y, rf.predict(val_xs))
```


# Bulldozers
now lets explore the importance of random forests and some data cleaning processes in fastai

## Data cleaning
first of, date columns. we can handle date columns using fastai's function `add_datepart(df, col name)`
```python
df = add_datepart(df, 'saledate')

' '.join(o for o in df.columns if o.startswith('sale'))
```
op:
```
'saleWeek saleYear saleMonth saleDay saleDayofweek saleDayofyear saleIs_month_end saleIs_month_start saleIs_quarter_end saleIs_quarter_start saleIs_year_end saleIs_year_start saleElapsed'
```
these are all the new columns added so that the random forest can be more flexible while splitting.
last time we had to manually change the datatypes as categorical. however now we can just use `TabularPandas` this is a class in fastai used for transforming.
we'll use two `TabularProc`s one is `Categorify`, which transforms a column into a numerical categorical column.
the other is `FillMissing`, which fills all the missing values with the median of the column. it also sets true in a column that indicates whether that row had a missing value.
`TabularPandas` also handles the training and validation split for us.
in most cases just choosing a random subset of your data as your validation dataset will do. however, here, since its a time series dataset lets make it so that the validation dataset has future dates.
for this data lets split on November 2011. the data containing dates after will be part of the validation dataset.
```python
cond = (df.saleYear<2011) | (df.saleMonth<10) 
train_idx = np.where( cond)[0] 
valid_idx = np.where(~cond)[0] 
splits = (list(train_idx),list(valid_idx))
```
`np.where()` returns the indices of all the data that are true for the given condition as the first element.
another thing to note is that `TabularPandas` needs to be told which are categorical data and which are continuous. so we use fastai's `cont_cat_split()` function.
the `cont_cat_split()` takes in a parameter named `maxcard` this parameter decides whether a column is categorical or not based on the number of unique values in the column. if the unique values are greater than `maxcard`, it is deemed a continuous column else a categorical. 
```python
cont,cat = cont_cat_split(df, 1, dep_var=dep_var)

to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
```
now we have our dataset. fastai's dataset has the attributes train and valid to see the data
also both train and valid has two attributes xs(independent) and y(dependent)
```python
len(to.train),len(to.valid)
to.show(3) -> behaves like showbatch
to.items.head(3) -> items return the original dataframe
```

the categories are label encoded just like any other usual categorical encoding, unless you set an ordinal category before using `categorify`. then the order you chose will be used.

## The Decision Tree
```python
xs,y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=4) 
m.fit(xs, y)
draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)
```
![[Pasted image 20231215220657.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215220657.png/?raw=true)

the nodes drawn from fastai's `draw_tree` function shows different values.
the first one is the column to do the split on. The second is the mean squared error of a model when that node is implemented. that means the first node is just a model that predicts the dependent variable to be the average of all the dependent variables and its `mse` is 0.48
the third is the number of samples in that group
and the final is the average of all the dependent variables in that group.
that is how random forests predict continuous variables. it averages the dependent variables in the group that the row belongs to and gives it as the prediction.

using the `detreeviz` library for visualizing gives us:
![[Pasted image 20231215221800.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215221800.png/?raw=true)
here we can see that there are some years that are marked as 1000! this is probably as missing value code.(used for filling in missing values while collecting data)
even if we set all those years to 1950 an do the decision tree again, there is little to no change in the decision tree. which shows that decision trees are resilient to data problems

now, if you don't set a limit for the `max_leaf_nodes`, the models gonna overfit by setting a node for each specific row. so we need to choose a good limit to stop the decision Tree
### Categorical variables
we have many ways to transform categorical variables like entity embeddings in [[Collaborative Filtering]] and one-hot encoding. however, decision trees don't really need any of that because, it doesn't need to be transformed. we just need a condition to do the split on and if the categorical columns are numerical then that's it. we don't need one hot encoding here as it will just consume extra space and it was shown that one hot encoding does **not** improve the decision tree's performance.

## Random Forests
as last time we use bagging to create multiple models and average them out to lower the errors and predict things closer and closer to the true value.
```python
def rf(xs, y, n_estimators=40, max_samples=200_000, 
	   max_features=0.5, min_samples_leaf=5, **kwargs):
	 return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
	  max_samples=max_samples, max_features=max_features,
	min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
```
this is a function to create random forests. 
`n_estimators` is the number of trees you want
`max_samples` is the number of rows to randomly subset from the original data. 
`max_features` is the number of columns to randomly subset from the original data.
`min_sample_leaf` is the number of leaf nodes we need at the end.
```python
m = rf(xs, y);
```
one of the main properties of random forests is that they are not that much sensitive to their hyper parameters.
`n_estimators` can be as high as you want as long as you have the time to train it.
![[Pasted image 20231215224151.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215224151.png/?raw=true)
from this graph here, we can see that as `n_estimators` increases the error rate always decreases. 
the blue line shows the random forests that subset very little columns/features from the original dataset(the square root of the number of features)
the green line takes no subset but rather all the features for all the trees in the forest. as you can see, using subsets of columns too gives better results.

to see the relation between the number of estimators and the error, lets take the prediction of each estimator(tree) and see how the average changes as we add more trees
```python
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
plt.plot([r_mse(preds[:i+1].mean(0), valid_y) for i in range(40)])
```
`m.estimators_` gives each tree.
so the plot shows the error for the forest with 1 tree till forest till 40 trees![[Pasted image 20231215224803.png]]
as you can see, the improvement goes on till about 30 trees where it levels of.
however, we see that the performance on our valid set is a little bit worse than the performance on the training set. it might be due to overfitting or due to this being a time series data. to find out we use the out-of-bag error
### The out-of-bag error
the OOB error is a way to measure prediction error based on the valid data of all the individual trees. since the trees used random subsets to split on, the subsets from the data not included in this will be the validation dataset. this allows us to see if our model is overfitting or if its any other error
```python
r_mse(m.oob_prediction_, y)
```
this gives us 0.21 which is way lesser than the error we got through the normal way. so that means our model is not overfitting and there is some other problem.

## Model Interpretation
Random forests are mainly used for their interpretations of the data.
there are many questions that can be answered by these interpretations

### Tree variance for Prediction confidence
to see how confident we are about a prediction, we can use standard deviation to see how spread out the predictions are for each row and if the spread is high, many trees disagree with the prediction, while if the spread is low, the confidence is high.
to implement this,
```python
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])

preds_std = preds.std(0)

preds_std[:5] -> returns the confidence for first 5 rows
```
lets say there are 100 rows in the valid data to predict. we use the 40 trees to predict 100 values each for all these rows.
so the shape of the `preds` should be [40, 100], there being 40 trees and each tree having 100 predictions.
now if we apply standard deviation across the rows we get the measure of how spread out the predictions of all the trees are for one row. this can be useful to check for auctions and you can check in on the low confidence predictions to decide whether to bid on them or not.

### Feature Importance
we can get which features are important for this model and for the data as a whole using the `feature_importances_` attribute.
```python
def rf_feat_importance(m, df): 
return pd.DataFrame({'cols':df.columns, 
					 'imp':m.feature_importances_} ).sort_values('imp', 
					 ascending=False)
fi = rf_feat_importance(m, xs) 

def plot_fi(fi): 
	return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False) 
plot_fi(fi[:30]);
```
a plot of the data frame of all the columns and their importance can give us perspective on which columns are more useful than others.
![[Pasted image 20231215230942.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215230942.png/?raw=true)

these importance are calculated by going through each tree and seeing each split to see what they split on and how much the error has improved based on the split. all of these are added to the importance factor of each column and finally they are all normalized so that they add up to 1.

### Removing low importance features
from the plot above we can see that some features have very less importance that they can be dropped.
this can be the first step for any problem because this simplifies the data and lets us focus on the important data. this also makes the feature importance plot more readable
```python
to_keep = fi[fi.imp>0.005].cols
xs_imp = xs[to_keep] 
valid_xs_imp = valid_xs[to_keep]
m = rf(xs_imp, y)
plot_fi(rf_feat_importance(m, xs_imp));
```
![[Pasted image 20231215231609.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215231609.png/?raw=true)
the data has been simplified from 78 columns to 21. this is really useful for us to focus on other problems

## Redundant columns
```python
cluster_columns(xs_imp)
```
![[Pasted image 20231215231844.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231215231844.png/?raw=true)
this plot shows us how similar some features are. features closely related or redundant ones tend to merge at the beginning, like `ProductGroup` and `ProductGroupDesc`. these type of features can be removed to further simplify our model.
this similarity is calculated by assigning a rank to all the values and then calculating the correlation. closely correlated ranks merge quickly.
now lets try removing these redundant features
```python
def get_oob(df): 
	m = RandomForestRegressor(n_estimators=40, min_samples_leaf=15, 
	max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True) 
	m.fit(df, y) 
	return m.oob_score_
```
this function is used to quickly return the `oob_score` of a data frame passed. `oob_score` is used to see if a random forest is good or bad.  The OOB score is a number returned by sklearn that ranges between 1.0 for a perfect model and 0.0 for a random model.
this is used to compare how the error changes for each change we make in the data.
```python
{c:get_oob(xs_imp.drop(c, axis=1)) for c in ( 'saleYear', 'saleElapsed',
		'ProductGroupDesc','ProductGroup', 'fiModelDesc', 'fiBaseModel',
		 'Hydraulics_Flow','Grouser_Tracks', 'Coupler_System')}
```
here we check the `oob score` after each removal
```
{'saleYear': 0.8766429216799364,
 'saleElapsed': 0.8725120463477113,
 'ProductGroupDesc': 0.8773289113713139,
 'ProductGroup': 0.8768277447901079,
 'fiModelDesc': 0.8760365396140016,
 'fiBaseModel': 0.8769194097714894,
 'Hydraulics_Flow': 0.8775975083138958,
 'Grouser_Tracks': 0.8780246481379101,
 'Coupler_System': 0.8780158691125818}
```
now lets drop multiple features. we'll drop one of the elements tightly related pairs.
```python
to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 
		   'Grouser_Tracks'] 
get_oob(xs_imp.drop(to_drop, axis=1))
```
we got 0.87 which is pretty good.
```python
xs_final = xs_imp.drop(to_drop, axis=1) 
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)
```
now we have simplified our data to just the important features.

## Partial Dependence
partial dependence is used to see how one independent variable affects the dependent variable, which is the Sales price here.
partial dependence answers the question, if a row varied on nothing but the the feature in question, how will the dependent value change?
here the two important features from our feature importance plot are the year made and product size.
to find the partial dependence of year made, we need to find how a row varies when we change the year made. we can't just average the predictions of rows with the same year made for this.
instead we replace all the values of year made, with 1950 and calculate predictions and then average them. same for the next year and so on.
then we can plot these averages against the years and *voila* - partial dependence plots
```python
from sklearn.inspection import plot_partial_dependence 
fig,ax = plt.subplots(figsize=(12, 4)) 
plot_partial_dependence(m, valid_xs_final, ['YearMade','ProductSize'],
						grid_resolution=20, ax=ax)
```
![[Pasted image 20231216154737.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231216154737.png/?raw=true)

the year made plot shows us that that the average price increases as the year increases. remember that the price predicted is taken after log. so it increases exponentially.
the product size plot shows that the prices are low for the last group, which is what we designated for missing values. we need to find out why it's missing

## Data Leakage
Data leakage is used when a model supposedly gets the target or, parameters that are not supposed to be known to the model as parameters.
for example, sometimes when a model is doing too good, it might be a good idea to check out the feature importance plots and use to to plot the partial dependence plots on the important ones.
simple approaches to identifying data leakage are to build a model and then:
- Check whether the accuracy of the model is _too good to be true_.
- Look for important predictors that don't make sense in practice.
- Look for partial dependence plot results that don't make sense in practice.

## Tree interpreters
for a particular row, to see which features are the important factors for predicting, we use tree interpreter. what this does is that, just like feature importance, we go through each tree and see the splits and add up the contributions to the importance variable.
we do this but for only 1 row. we put the row we want through the first tree, see how the features change the prediction and add up to each feature. then we add this up for all trees.
```python
from treeinterpreter import treeinterpreter 
from waterfall_chart import plot as waterfall

row = valid_xs_final.iloc[:5]
prediction,bias,contributions = treeinterpreter.predict(m, row.values)
```
prediction is the prediction of the forest. bias is the mean of all the dependent variables of all the trees. and contributions are the contributions of each feature to see how the bias changes to the prediction. that means bias plus the sum of the contributions of all features gives us the prediction.
plotting the contributions give us on what basis did each column affect the dependent variable
we use waterfall plots for these
```python
waterfall(valid_xs_final.columns, contributions[0], threshold=0.08,
		  rotation_value=45,formatting='{:,.3f}');
```
![[Pasted image 20231216164353.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231216164353.png/?raw=true)

these type of plots are used for production rather than model development.

## The Extrapolation Problem
extrapolation problem is that random forests can't predict values outside of the range of dependent variables given while training. this is because, random forests just give out the average of the predictions of trees and trees only give out the average of the dependent variable in a leaf. so this means the predicted value can never be higher than the maximum dependent variable in the training set.
![[Pasted image 20231216165028.png]] ![[Pasted image 20231216165047.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231216165028.png/?raw=true)
the red ones are predictions and the blue ones are actual values.
this is particularly a problem with datasets like inflation and stock changes.
so we have to make sure the validation dataset doesn't contain any out-of-domain data.
### Finding out-of-domain data
to find out of domain data we can use another random forests that predicts whether a row is from the training dataset or the valid dataset.
when we build a model like that and plot the feature importance of that model, we can see which features are very important for distinguishing between training dataset and valid dataset. that means these columns might be the reason for the out of domain problem
so then we remove these columns
```python
df_dom = pd.concat([xs_final, valid_xs_final]) 
is_valid = np.array([0]*len(xs_final) + [1]*len(valid_xs_final)) 
m = rf(df_dom, is_valid) 
rf_feat_importance(m, df_dom)[:6]
```
![[Pasted image 20231216165912.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231216165912.png/?raw=true)
we can see that `saleElapsed` which is directly connected to the time elapsed, `SalesID` and `MachineID` have extreme importance for seeing which rows are from the test or valid set.
so now we try removing them
```python
m = rf(xs_final, y) 
print('orig', m_rmse(m, valid_xs_final, valid_y)) 
for c in ('SalesID','saleElapsed','MachineID'): 
	m = rf(xs_final.drop(c,axis=1), y) 
	print(c, m_rmse(m, valid_xs_final.drop(c,axis=1), valid_y))
```
```
orig 0.232883
SalesID 0.230347
saleElapsed 0.235529
MachineID 0.230735
```
this means we can safely remove `salesID` and `machineID` because they don't affect the error that much.
```python
time_vars = ['SalesID','MachineID'] 
xs_final_time = xs_final.drop(time_vars, axis=1) 
valid_xs_time = valid_xs_final.drop(time_vars, axis=1) 
m = rf(xs_final_time, y) 
```
Removing these variables has slightly improved the model's accuracy; but more importantly, it should make it more resilient over time

## Neural Networks
lets try a neural network for this data. recall how it works in [[Neural Networks]]
```python
df_nn = pd.read_csv(path/'TrainAndValid.csv', low_memory=False) df_nn['ProductSize'] = df_nn['ProductSize'].astype('category') df_nn['ProductSize'].cat.set_categories(sizes, ordered=True,
					inplace=True) 
df_nn[dep_var] = np.log(df_nn[dep_var]) 
df_nn = add_datepart(df_nn, 'saledate')
```
we can use the simplified columns we got from the random forests
```python
df_nn_final = df_nn[list(xs_final_time.columns) + [dep_var]]
```
in neural networks categorical variables are handled like we will see in [[Collaborative Filtering]]. we use entity embeddings for neural networks.
```python
cont_nn,cat_nn = cont_cat_split(df_nn_final, max_card=9000, 
								dep_var=dep_var)

df_nn_final[cat_nn].nunique()
YearMade                73
ProductSize              6
Coupler_System           2
fiProductClassDesc      74
Hydraulics_Flow          3
ModelID               5281
fiSecondaryDesc        177
fiModelDesc           5059
Enclosure                6
Hydraulics              12
ProductGroup             6
Drive_System             4
Tire_Size               17
dtype: int64
```
we can see that the `modelID` and `fiModelDesc` have similar number of values and might be redundant. since they are categorical we don't want to be creating above 5000 columns for both embeddings so we check with a random forest to see if it is ok with deleting one.
```python
xs_filt2 = xs_filt.drop('fiModelDescriptor', axis=1) 
valid_xs_time2 = valid_xs_time.drop('fiModelDescriptor', axis=1) 
m2 = rf(xs_filt2, y_filt) 
m_rmse(m2, xs_filt2, y_filt), m_rmse(m2, valid_xs_time2, valid_y)
```
the error doesn't change that much so we can go ahead and delete them. and use `TabularPandas`
```python
cat_nn.remove('fiModelDescriptor')

procs_nn = [Categorify, FillMissing, Normalize] 
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, 
					  splits=splits, y_names=dep_var)
dls = to_nn.dataloaders(1024)
```
for regression models its good to set the range of values we want like in [[Collaborative Filtering]]
so
```python
y = to_nn.train.y 
y.min(),y.max()
(8.465899467468262, 11.863582611083984)
```
now lets create the neural network
```python
learn = tabular_learner(dls, y_range=(8,12), layers=[500,250],
						n_out=1, loss_func=F.mse_loss)
learn.lr_find()
```
![[Pasted image 20231216172205.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%206/Attachments/Pasted%20image%2020231216172205.png/?raw=true)
this is not a pre trained model so we use `fit_one_cycle`
```python
learn.fit_one_cycle(5, 1e-2)
preds,targs = learn.get_preds()
```
and that's the neural network

## Ensemble
since the neural network and random forest are two different models, we can ensemble them both to give a good prediction
```python
rf_preds = m.predict(valid_xs_time) 
ens_preds = (to_np(preds.squeeze()) + rf_preds) /2
```
we squeeze the predictions of the neural network into an array of the same size as the predictions of the random forest.
this gives us the best score so far.

## Boosting
there is something called gradient boosting, which is done as follows
- Train a small model that underfits your dataset.
- Calculate the predictions in the training set for this model.
- Subtract the predictions from the targets; these are called the "residuals" and represent the error for each point in the training set.
- Go back to step 1, but instead of using the original targets, use the residuals as the targets for the training.
- Continue doing this until you reach some stopping criterion, such as a maximum number of trees, or you observe your validation set error getting worse.

unlike random forests we can definitely overfit with boosting. because the more trees we have the lesser the error becomes and it finally overfits


