# word_prediction model

| Epoch      | Validation loss | Validation Accuracy     |
| :---        |    :----   |          :---: |
| 1      |  3.42      | 30%   |
| 2   |   3.14      |   33%    |
| 3      |    2.98    |  36%  |
| 4   |   2.77      |   38%    |
| 5      |    2.71    |  39%  |
| 6   |    2.42     |   43%    |
| 7      |  2.36      | 44%   |
| 8   |    2.35     |    45%   |

================================================================
Please run word_prediction.py to train model. 

Specify train data location with path_train_data. For example './data/train.csv'

Specify location for model checkpoints (in between epochs) with model_save_checkpoint. For exmaple './data/ft_xlm_model'

Trained model is also available with me which I can zip and send.

Default hyper-parameters in the code train well. 

================================================================

For generating test results, please run test.py.

Specify trained model path with path_to_model. For example './data/final_xlm_model'

Specify path to test dataset with path_to_test_data. For example './data/test.csv'

================================================================

data_read.py contains Data Loader. Currently sampling fraction of data set is set at 1,
but can be changed between 0-1 values.

model.py contains code for the model, which is an XLM-Roberta Model , using the XLM-Roberta tokenizers.
Pre-trained model taken from hugging face and then trained using the 6 choice blank word prediction task.
