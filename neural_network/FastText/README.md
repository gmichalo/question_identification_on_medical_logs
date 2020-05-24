# FastText

This is an implementation for Bag of Tricks for Efficient Text Classification [FastText](https://arxiv.org/pdf/1607.01759.pdf)

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials) run :
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name fasttext.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "FastText"  -fn "FastText" - -jf "search_spaces/fasttext.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/fasttext.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "FastText"  -fn "FastText"
```
