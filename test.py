import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
from sklearn import metrics
import transformers

#prepare test data 

def lower_str(s):
        return s.lower()


test_comments = pd.read_csv('data/Test.csv', sep = '\t', header = None)
test_comments.columns = ['comments']
test_comments['comments'] = test_comments['comments'].apply(lower_str)
test_labels = pd.read_csv('data/Test_label.csv', sep = '\t', header = None)
test_labels.columns = ['labels']

sentences = test_comments.comments.values
labels = test_labels.labels.values

#load model and tokenizer
output_dir ='model_save'
model = transformers.BertForSequenceClassification.from_pretrained(output_dir, 
                                                                   num_labels = 2, 
                                                            output_attentions = False,
                              output_hidden_states = False,)
tokenizer = transformers.BertTokenizer.from_pretrained(output_dir, do_lower_case=True)



# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to('cpu') for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions.
      result = model(b_input_ids, 
                     token_type_ids=None, 
                     attention_mask=b_input_mask,
                     return_dict=True)

  logits = result.logits

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)


#binarize the real values
pred_labels = []
for i in range(len(true_labels)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  pred_labels.append(pred_labels_i)

#flatten
bert_gold = np.concatenate(true_labels).ravel()
bert_pred = np.concatenate(pred_labels).ravel()

#store results
res_output_path = './results/'
res_df = test_comments.copy()
res_df['gold_labels'] = bert_gold
res_df['DynBERT_pred'] = bert_pred
res_df.to_csv(res_output_path + 'pred_results.csv', sep = '\t')

#write f1 score
f1_score = metrics.f1_score(bert_gold, bert_pred, average='macro')
with open('results/f1_result.txt', 'w') as f:
    f.write('The Macro F1 score of the DynBERT model tested on Dynamic test data: ' + str(f1_score))


print('    DONE.')
