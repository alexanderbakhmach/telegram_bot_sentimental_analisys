# Tweets sentiment analysis bot


Telegram bot which analyze typed text. There are 3 category that bot can attached to analyzed text
- Positive sentence
- Negative sentence
- Neutral sentence

#### Usage:
 Type some message to your bot ant hr will response to you with the message with probability of each sentence type and provide the char illustrated probabilities.

#### Instalation:
Install all required packages. Use python3 stable version
```
$ pip3 install -r requirements.txt
```
When all requirement are satisfied generate train data with [generator](https://github.com/alexanderbakhmach/twitter_dataset_adapter.git) or unzip already prepared generated data.
```
$ unzip generated/data.zip
```
Now lets`s train our neural network
```
$ chmod +x train.sh
$ ./train.sh
```
It will save trained **.h5** model with name **"model"** to the **"./generated"** folder

At the last step we need to edit our config.json file and check that all data there is correct
#### Running
To run the but execute:
```
$ python3 bot.py
```
