from transformers import BertModel
from transformers.models.bert.modeling_bert import BertConfig
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
import numpy as np

class BertEmbedder:
    #from https://gist.github.com/shubhamagarwal92/37ccb747f7130a35a8e76aa66d60e014
    def __init__(self,
                 pretrained_weights='bert-base-uncased',
                #  tokenizer_class=BertTokenizer,
                 model_class=BertModel,
                 max_seq_len=8,
                 config=None,
                 pretrained=True,
                 device=None):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        # self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_config = BertConfig(
            vocab_size=config["vocab_size"],#len(self.num_objs)+len(self.num_attrs),#len(dset.pairs),#self.config["vocab_size"],#self.num_objs,#
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        if pretrained:
            self.model = self.model_class.from_pretrained(pretrained_weights)
        else:
            self.model = self.model_class(self.bert_config)
        self.model = self.model.to(self.device)
        self.max_seq_len = max_seq_len
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)
        self.tokenizer = self.get_pretrained_tokenizer(pretrained_weights)
        self.vocab_size = self.tokenizer.vocab_size
        
    def get_pretrained_tokenizer(self,from_pretrained):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, do_lower_case="uncased" in from_pretrained
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, do_lower_case="uncased" in from_pretrained
        )

    def get_text(self, text):
        #index, caption_index = self.index_mapper[raw_index]

        # text = self.all_texts[raw_index]#[caption_index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_special_tokens_mask=True,
        )

        return (text, encoding)

    def tokenize_texts(self, texts):
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def get_bert_embeddings(self,
                            input_ids,mask):
        # examples = create_examples(raw_text)

        # features = convert_examples_to_features(
        #     examples, self.tokenizer, self.max_seq_len, True)

        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        if torch.is_tensor(input_ids):
            input_tensor = input_ids.to(self.device)
        else:
            input_tensor = torch.from_numpy(np.asarray(input_ids)).to(self.device)

        if torch.is_tensor(mask):
            mask_tensor = mask.to(self.device)
        else:
            mask_tensor = torch.from_numpy(np.asarray(mask)).to(self.device)
        
        # If it's a single sentence (1D array), we need to unsqueeze to make it (1, seq_len)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
            
        last_hidden_states = self.model(input_tensor, attention_mask=mask_tensor).last_hidden_state  
        
        # Return the [CLS] token representation for all items in the batch
        return last_hidden_states[:, 0, :]