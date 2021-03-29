class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        self.encoder = BertForMultipleChoice.from_pretrained(options_name,num_labels=4)
    def forward(self, model_input, labels):
        enc_output = self.encoder(model_input, labels=labels)
        loss, text_fea = enc_output[:2]
        return loss, text_fea