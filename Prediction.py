from transformers import AlbertTokenizer,BertTokenizer
import torch
class Prediction:
  def __init__(self, model, tweet):
    self.model = model
    self.tweet = tweet

  def myfunc(self):
      map_location = torch.device('cpu')
      print("in")

      test_str= self.tweet
      #print(test_str)
      if self.model=='BERT':
          saved_model = torch.load('model_bert', map_location='cpu')
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

      if self.model=='ALBERT':
          saved_model = torch.load('model_albert', map_location='cpu')
          tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

      encoded_dict_test = tokenizer.encode_plus(
          test_str,  # Sentence to encode.
          add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
          max_length=100,  # Pad & truncate all sentences.
          padding='max_length',
          return_attention_mask=True,  # Construct attn. masks.
          return_tensors='pt',
          truncation=True,  # Return pytorch tensors.
      )
      #print(encoded_dict_test)
      #print("------------")
      saved_model.eval()
      result_test = saved_model(encoded_dict_test['input_ids'],
                                token_type_ids=None,
                                attention_mask=encoded_dict_test['attention_mask'],
                                labels=None,
                                return_dict=True)
      logits = result_test.logits
      #print(logits)
      a = logits[0].argmax().cpu().numpy()
      print(a)
      if a==0:
          return 'Neutral'
      if a==1:
          return 'Positive'
      if a==2:
          return 'Negative'


