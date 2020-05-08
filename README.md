# BiLSTM_CNN:
<br>Requirements:
<br>Python==3.X.X
<br>Tensorflow==1.13.1
<br>Keras==2.1.6
<br>scipy==1.4.1

To run the model:
<br> python BiLSTM-CNN/main.py

# BiLSTM-Attention:
<br>Requirements:
<br>Python==3.X.X
<br>Tensorflow==1.13.1
<br>Keras==2.1.6
<br>scipy==1.4.1

To run the model:
<br> python main.py
( Change main to train() first, and then predict())

# Bert Model:
pandas==0.24.2
torch==1.2.0
numpy==1.16.4
transformers==2.0.0

# Fine Tuning BERT


Run the following command to install all required packages:
```
pip install -r requirements.txt
```
Download Fine Tune Bert from google drive link: https://drive.google.com/open?id=1FZJWAlyT7_IRN0wgp3WMYZvG1TI7bjff
Donwload Bert VIZ from google drive link: https://drive.google.com/open?id=1KSYhxvqqcl6NAiZbDMG3jiRVFY3WU08p
```
## Data
Is provided in data folder (Sample data)

## Training the model
If you want to train the model uncomment train and validation data in main.py and comment out the loading of the model. If you just want to test it use the saved model and uncomment the loading and comment the train and test data.
Download the train data and the saved model from 

To train the model with fixed weights of BERT layers, execute the following command from the src directory
```
python -m main -freeze_bert -gpu <gpu to use> -maxlen <maximum sequence length> -batch_size <batch size to use> -lr <learning rate> -maxeps <number of epochs>
```
To train the entire model i.e. both BERT layers and the classification layer just skip the -freeze_bert flag
```
python -m main -gpu <gpu to use> -maxlen <maximum sequence length> -batch_size <batch size to use> -lr <learning rate> -maxeps <number of epochs>
```

# BertViz:
The three IPYNB mentioned is all you need to run. 

Referenced from: https://github.com/jessevig/bertviz
