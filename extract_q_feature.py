import numpy as np
import pandas as pd
import os
import json
#import ipdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection





# loaded_npz = np.load('/home/conda-user/workspace_jinu/features/clip_text_features/qid2.npz')
# keys : last_hidden_state, pooler_output

train_path = '/home/conda-user/workspace_jinu/QD-DETR/data/highlight_train_release.jsonl'
val_path = '/home/conda-user/workspace_jinu/QD-DETR/data/highlight_val_release.jsonl'  
re_val_path = '/home/conda-user/workspace_jinu/QD-DETR/data/highlight_val_release_val3.jsonl' # 0.7
# up_re_val_path = '/home/conda-user/workspace_jinu/QD-DETR/data/updated_highlight_val_release.jsonl' # 0.3
test_path = '/home/conda-user/workspace_jinu/QD-DETR/data/highlight_test_release.jsonl'

# ipdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


query_list = []
# ipdb.set_trace()
query_df = pd.read_json(train_path, lines=True)

# 누적해서 더하는 식으로
# qid change
# query_df['qid'] = query_df['qid'].apply(lambda x : x+20000)
    
#/home/jejekim/VTimeLLM/feature_qvh/clip_text_features_val3
for i, query in tqdm(enumerate(query_df['query'])):
    with torch.no_grad():
        inputs = tokenizer(query, padding=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        # ipdb.set_trace()
        
        output = outputs.last_hidden_state.squeeze().cpu().numpy()
        output2 = outputs.text_embeds.squeeze().cpu().numpy()
        save_path = '/home/jejekim/VTimeLLM/feature_qvh/clip_text_features_newval/'
        save_name = "qid" + str(query_df['qid'].iloc[i]) + '.npz'
        np.savez(save_path+save_name, last_hidden_state=output, pooler_output=output2)

print('Done')

# find /home/jejekim/workspace_jinu/QD-DETR/features/clip_text_features -type f -regex '.*/qid2[0-2][0-9][0-9][0-9][0-9]\.npz' -print
