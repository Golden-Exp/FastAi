#FastAi #tabular 
A *language model* is a model that predicts the next word of a sentence given.
training for this is by a method named *self supervised learning* -> that is the model learns on its own with the inputs given(which is just lots of text) and doesn't need labels from us. instead it takes the labels embedded in the input.
### ULMFit
Universal Language Model Finetuning. normally, in NLP, we use a pretrained language model that can predict the next word of an English sentence. that model is finetuned to become a classifier for whatever use we want.
however in ULMFit, after getting the pretrained model, we then finetune the model on the dataset we have and are going to use for our NLP model. for example if we want to classify IMDB reviews, we get a language model pretrained on Wikipedia, finetune it on the IMDB reviews to get a language model that can predict the next word of an IMDB review. 
**Then** we use that model for the required task , i.e. classifier
![[Pasted image 20231109184538.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231109184538.png/?raw=true)
this is the ULMFit approach
### basic steps of NLP
- Tokenization: to convert the sentence into a list of words -> this list of words is known as vocab.
- Numericalization: to feed into neural networks we need numbers and not string. so we convert our tokens into numbers, i.e. the indices of the tokens in the list
- Language model Data Loader: their is a separate dataloader for language models in fastai known as `LMDataloader` . this is a convenient Dataloader mainly for nlp in fastai
- Language model: next we need the model itself

now each steps are:
#### Tokenization
there are many approaches for tokenization since there many things to consider in an English sentence like punctuation and stuff. Do we split on words? if so what about words like "back-to-back"?
3 main approaches of tokenization are:
- Word based -> split on spaces, punctuations are separate
- Sub-word based -> split on words and sub-words
- Character based -> split on each character
tokenization with fastai
##### word tokenization
```python
#getting the dataset
from fastai.text.all import * 
path = untar_data(URLs.IMDB)

#setting up the path
files = get_text_files(path, folders = ['train', 'test', 'unsup'])

#sample of the data  
txt = files[0].open().read()

#wordtokenizer points to the deafault tokenizer of fastai which us spacy here
spacy = WordTokenizer()     

#tokenizing text and printing the first few
toks = first(spacy([txt]))
print(coll_repr(toks, 30))
```
output:
```
(#201) ['This','movie',',','which','I','just','discovered','at','the','video','store',',','has','apparently','sit','around','for','a','couple','of','years','without','a','distributor','.','It',"'s",'easy','to','see'...]
```
`coll_repr(collection, n)` -> prints the first n elements of the collection
tokenizer takes a collection as input to tokenize. so we convert txt to a list before giving it to spacy

now when we give spacy to the `Tokenizer` class of fastai and use that to get the tokenized content
```python
tkn = Tokenizer(spacy) 
print(coll_repr(tkn(txt), 31))
```
output:
```
(#228) ['xxbos','xxmaj','this','movie',',','which','i','just','discovered','at','the','video','store',',','has','apparently','sit','around','for','a','couple','of','years','without','a','distributor','.','xxmaj','it',"'s",'easy'...]
```
we have extra things with "xx" as a prefix. those are known as _special tokens_
example: `xxbos` means that is the beginning of the sentence. this can tell the model that this is a new sentence and the model will act accordingly.
this makes the model recognize important parts of a text. It also helps the model save memory
some main special tokens are:
- `xxbos`: indicates the beginning of the sentence
- `xxmaj` : indicates that the next word starts with a capital letter
- `xxunk`: the word is unknown
`defaults.text_proc_rules` -> used to see the rules for all special tokens
some of the rules are:
![[Pasted image 20231109192332.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231109192332.png/?raw=true)

##### Sub-word Tokenization
used in different languages where the concept of a "word" is blurry like Chinese and Japanese.
to do Sub-word tokenization we observe a corpus of the text and find out the most commonly occurring groups of letters. now then we tokenize this.
```python
txts = L(o.open().read() for o in files[:2000])
#L is the same as coll_repr

def subword(sz): 
	sp = SubwordTokenizer(vocab_sz=sz) 
	sp.setup(txts) 
	return ' '.join(first(sp([txt]))[:40])

subword(1000)
```
output:
```
'▁This ▁movie , ▁which ▁I ▁just ▁dis c over ed ▁at ▁the ▁video ▁st or e , ▁has ▁a p par ent ly ▁s it ▁around ▁for ▁a ▁couple ▁of ▁years ▁without ▁a ▁dis t ri but or . ▁It'
```

here `setup` is used to "train" the tokenizer to find the vocab for the sub-words. Also when using sub-words ' _ ' denotes a space.
the number and the tokens themselves change with the vocab size. if a larger size is given, multiple words together can form a token.
if a smaller size is given then, even characters form tokens.

**_Picking a sub-word vocab size represents a compromise: a larger vocab means fewer tokens per sentence, which means faster training, less memory, and less state for the model to remember; but on the downside, it means larger embedding matrices, which require more data to learn.**_

#### Numericalization
- make the vocab
- replace all the elements in the vocab with their index in the vocab

```python
#tokenizing first 200 words in txts
toks200 = txts[:200].map(tkn) 

#creating the numericalizer and using setup to create the vocab
num = Numericalize() 
num.setup(toks200) 
coll_repr(num.vocab,20)

```
output:
```
"(#2000) ['xxunk','xxpad','xxbos','xxeos','xxfld','xxrep','xxwrep','xxup','xxmaj','the','.',',','a','and','of','to','is','in','i','it'...]"
```
just like during sub-word tokenizer we use setup to get the vocab.

the special tokens appear first and then the others. `Numericalize()` has a default `max_vocab` as 60000. so, aside from the most common 60000 elements, the remaining elements will be `xxunk

also the default value of `min_frequency` is 3. so only if a word is repeated 3 or more times is it allowed to be in the vocab. else it will also be replaced with `xxunk`

```python
#using the numericalize function
nums = num(toks)[:20]; 
nums
```
output:
```
tensor([  2,   8,  21,  28,  11,  90,  18,  59,   0,  45,   9, 351, 499,  11,  72, 533, 584, 146,  29,  12])
```

thus all elements were numericalized.
#### Separating into batches

to give the data to the model, the order of the tokens has to be the same and can't be changed. so even while separating into batches, we need to give the right order.

we first divide the data given by the bs.
then we divide the whole array of all the divisions into sub-arrays to feed to the machine.

so that means the second sequence in a sub-array is not a continuation of the first sequence. it is an entirely new one. however the first sequence of the next sub-array **should be a continuation of the first sequence in the previous batch**. 

this enables the model to learn since all these are happening parallelly. that is, each sequence of a batch is being read parallelly and not one after the other.

suppose these are the tokens that are present
```
xxbos xxmaj in this chapter , we will go back over the example of classifying movie reviews we studied in chapter 1 and dig deeper under the surface . xxmaj first we will look at the processing steps necessary to convert text into numbers and how to customize it . xxmaj by doing this , we 'll have another example of the preprocessor used in the data block xxup api . \n xxmaj then we will study how we build a language model and train it for a while .
```
now lets say we have a bs of 6. so we need to separate the data like
![[Pasted image 20231110003723.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110003723.png/?raw=true)
this is bs=6 and sequence length 15
for bs=6 and seq_len = 5
we divide the whole into 3
then into sub-arrays like this
![[Pasted image 20231110003744.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110003744.png/?raw=true)
and this
![[Pasted image 20231110003812.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110003812.png/?raw=true)
then finally this
![[Pasted image 20231110003832.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110003832.png/?raw=true)

now the stream of tokens that are divided into batches, should be a concatenation of all the reviews. this stream should then be divided into batches.

however it is necessary to randomize the entries for each epoch to learn more efficiently like with images.
so the order of concatenation is random for each epoch.

**_note that only the order of concatenating the reviews to get the "stream" is randomized for each epoch. The order of the tokens in each review itself is not randomized.**_

![[Pasted image 20231110001904.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110001904.png/?raw=true)

while concatenating it is a good idea to specify some extra text before each new text "element"
like when concatenating tweets or something, this can be useful
![[Pasted image 20231110090919.png]](https://github.com/Golden-Exp/FastAi/blob/main/Lesson%204/Attachments/Pasted%20image%2020231110090919.png/?raw=true)


all of this is done by `LMDataLoader`
```python
#numericalizing
nums200 = toks200.map(num)

dl = LMDataLoader(nums200)

#x is the independent variable and y is the dependent variable
x,y = first(dl) 
' '.join(num.vocab[o] for o in x[0][:20])
' '.join(num.vocab[o] for o in y[0][:20])
```
output
```
'xxbos xxmaj this movie , which i just xxunk at the video store , has apparently sit around for a'

'xxmaj this movie , which i just xxunk at the video store , has apparently sit around for a couple'
```

#### Language Model

```python
get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup']) 
dls_lm = DataBlock( blocks=TextBlock.from_folder(path, is_lm=True), 
				   get_items=get_imdb, splitter=RandomSplitter(0.1) 
				   ).dataloaders(path, path=path, bs=128, seq_len=80)
```
here DataBlock is used to make sure Tokenization and Numericalization are done automatically.
when the block is given as `TextBlock` this process is automatic.

now we have the Dataloader and as usual `dls.show_batch()` works the same.
now we need the learner to be the one pretrained on the wiki texts.

so the AWD_LSTM architecture is used.
```python
learn = language_model_learner(dls_lm, 
							   AWD_LSTM, 
							   drop_mult=0.3, 
							   metrics=[accuracy,Perplexity()]).to_fp16()

#training the frozen model
learn.fit_one_cycle(1, 2e-2)
```
by default the pretrained models are frozen. that is when we finetune only the random weights are changed. the fixed weights of the inner layers are not considered. so unfreeze is used to modify them according to the model we need.
```python
learn.unfreeze() 
learn.fit_one_cycle(10, 2e-3) -> 10 epochs
learn.save_encoder('finetuned') -> saving the model
```

```python
TEXT = "I liked this movie because"
N_WORDS = 40 
N_SENTENCES = 2 
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)]
```
output
```
i liked this movie because of its story and characters . The story line was very strong , very good for a sci - fi film . The main character , Alucard , was very well developed and brought the whole story
i liked this movie because i like the idea of the premise of the movie , the ( very ) convenient virus ( which , when you have to kill a few people , the " evil " machine has to be used to protect
```
this is text generation
#### Classification Data Loaders
```python
dls_clas = DataBlock(blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),
							 CategoryBlock), get_y = parent_label,
							 get_items=partial(get_text_files, 
							 folders=['train', 'test']), 
							 splitter=GrandparentSplitter(valid_name='test') ).dataloaders(path, path=path, bs=128, seq_len=72)
```
here the output is a category block so it means classification.

```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, 
								drop_mult=0.5, metrics=accuracy).to_fp16()

#loading the lm on the classifier
learn = learn.load_encoder('finetuned')
```

```python
learn.fit_one_cycle(1, 2e-2)

#freezing everything except last two layers
learn.freeze_to(-2) 
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

#freezing everything except last 3 layers
learn.freeze_to(-3) 
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

#unfreezing
learn.unfreeze() 
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```

**_We reached 94.3% accuracy, which was state-of-the-art performance just three years ago. By training another model on all the texts read backwards and averaging the predictions of those two models, we can even get to 95.1% accuracy, which was the state of the art introduced by the ULMFiT paper._**

