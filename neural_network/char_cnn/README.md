# CHAR_CNN

This is an implementation for Character-level Convolutional Network [CHAR_CNN](https://arxiv.org/pdf/1509.01626.pdf)

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials) run :
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name char_cnn.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "CHAR_CNN"  -fn "CHAR_CNN" - -jf "search_spaces/char_cnn.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/char_cnn.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "CHAR_CNN"  -fn "CHAR_CNN"
```
