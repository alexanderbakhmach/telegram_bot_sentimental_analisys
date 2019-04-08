# Tweets sentiment analysis bot


Telegram bot which analyze typed text. There are 3 category that bot can attached to analyzed text
- Positive sentence
- Negative sentence
- Neutral sentence

#### Usage:
 Type some message to your bot ant hr will response to you with the message with probability of each sentence type and provide the char illustrated probabilities.

#### Instalation:
The example to install project and configure it with the virtualenv
Install all required packages. Use python3 stable version
```
$ virtualenv -p python3 env
$ source env/bin/activate
$ pip install -r requirements.txt
```
When all requirement are satisfied generate train data with [generator](https://github.com/alexanderbakhmach/twitter_dataset_adapter.git) or unzip already prepared generated data.
```
$ cd generated
$ unzip generated/data.zip
$ cd ..
```
Now lets\`s train our neural network
All files we will store in generated folder
Let\`s name our model "model"
```
$ python train.py -s ./generated -d ./generated/data.csv -n model 
```
It will save trained **.h5** model with name **"model"** to the **"./generated"** folder

At the last step we need to edit our config.json file and check that all data there is correct
#### Running
To run the but execute:
```
$ python3 bot.py
```
