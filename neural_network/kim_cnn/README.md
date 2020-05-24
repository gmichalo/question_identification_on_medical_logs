# KIM_CNN

This is an implementation for Convolutional Neural Networks for Sentence Classification [KIM_CNN](https://arxiv.org/pdf/1408.5882.pdf)

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials):
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name kim_cnn.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "KIM_CNN"  -fn "KIM_CNN" - -jf "search_spaces/kim_cnn.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/kim_cnn.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "KIM_CNN"  -fn "KIM_CNN"
```
