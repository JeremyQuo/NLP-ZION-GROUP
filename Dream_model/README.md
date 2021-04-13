
The implementations follow the baseline system descriptions in the following two papers. 

* [Improving Question Answering with External Knowledge](https://arxiv.org/abs/1902.00993)
* [Probing Prior Knowledge Needed in Challenging Chinese Machine Reading Comprehension](https://arxiv.org/abs/1904.09679)

Here, we show the usage of this baseline using a demo designed for [DREAM](https://dataset.org/dream/), a dialogue-based three-choice machine reading comprehension task.

  1. Download and unzip the pre-trained language model from https://github.com/google-research/bert
  2. Set up the  variable in ```bert_path.py``` such as  ```bert_path='../../uncased_L-12_H-768_A-12/'```
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py```
  4. Execute ```python run_classifier.py  ```
  5. The resulting fine-tuned model, predictions, and evaluation results are stored in ```bert/dream_finetuned```.

**Results on DREAM**:

We run the experiments five times with different random seeds and report the best development set performance and the corresponding test set performance. 

| Method/Language Model | Batch Size | Learning Rate | Epochs | Test |
| --------------------  | ---------- | ------------- | ------ | ---- |
| BERT-Base, Uncased    | 8         | 2e-5          | 8      | 56.6 |


