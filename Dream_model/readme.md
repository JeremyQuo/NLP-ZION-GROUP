# Operation Manual: Model based on BERT (Dream)


## Requirement: 
Based on the size of dataset, a GPU whose memory is equal or larger than 8 GB is needed in this experiment. And the expected running time is about 40 minutes (running on Nvidia RTX 3060Ti)


### Environment: 

Python >= 3.6

PyTorch >= 1.0

## Usage

### Installation

- 1.	Download and unzip the pre-trained language model from https://github.com/google-research/bert. (in our experiment, I chose ```BERT-Base(Uncased)``` and ```BERT-Large(Uncased)```.
- 2.	Set the path of pre-trained language model in ```/Dream_model/bert/bert_path.py``` like below:
        ```python
        bert_path='../../uncased_L-12_H-768_A-12/'
        ```
- 3.	In ```/bert```, execute 
        ```python convert_tf_checkpoint_to_pytorch.py```, which can convert the format of TensorFlow to py-torch and save it as pytorch_model.bin
- 4.	Execute ```python run_classifier.py``` to train the model and evaluate the result.



### Result
**Results on DREAM**:

The hyper-parameter like random seeds and epoch_number can be tuned in ```run_classifier.py```. 

However, because of hardware’s limitation, the batch size and max sequence length can’t be larger, which limited the result.

| Method/Language Model | Batch Size | Learning Rate | Epochs | Test |
| --------------------  | ---------- | ------------- | ------ | ---- |
| BERT-Base, Uncased    | 8         | 2e-5          | 8      | 56.6 |
| BERT-Large, Uncased   | 8         | 2e-5          | 8      | 61.1 |

### Reference
The implementations follow the baseline system descriptions in the following two papers. 

* [Improving Question Answering with External Knowledge](https://arxiv.org/abs/1902.00993)
* [Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679)