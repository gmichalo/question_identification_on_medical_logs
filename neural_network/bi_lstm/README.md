# bi-lstm

This is an implementation of a bi-lstm network which is equivalent of the **quest-cnn** in the paper Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials) :
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name bi_lstm.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "BI_LSTM"  -fn "BI_LSTM" - -jf "search_spaces/bi_lstm.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/bi_lstm.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "BI_LSTM"  -fn "BI_LSTM"
```
