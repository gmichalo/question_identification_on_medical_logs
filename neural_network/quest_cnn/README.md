# Quest_CNN

This is an implementation for Where's the Question? A Multi-channel Deep Convolutional Neural Network for Question Identification in Textual Data

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials) :
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name quest_cnn.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "QUEST_CNN"  -fn "QUEST_CNN" - -jf "search_spaces/quest_cnn.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/quest_cnn.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "QUEST_CNN"  -fn "QUEST_CNN"
```
