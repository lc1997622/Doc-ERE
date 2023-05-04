import sys
sys.path.append("../")
sys.path.append("src/")
from src.utils import to_cuda
from src.model_prompt2 import Model
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data_prompt2 import get_dataloader,get_docloader,myDocDataset
from transformers import AdamW, RobertaTokenizerFast, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from src.data_prompt import REL2ID, ID2REL
import warnings
import os
import sys
from pathlib import Path 
from src.dump_result import temporal_dump
from torch.cuda.amp import autocast
from src.loss import focal_loss

warnings.filterwarnings("ignore")

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def evaluate(model, dataloader, doc_embeddings ,desc=""):
    global REPORT_CLASS_NAMES
    global REPORT_CLASS_LABELS
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            scores = model(data,doc_embeddings)
            labels = data["label"]
            pred = torch.argmax(scores, dim=-1)
            pred_list.extend(pred.cpu().numpy().tolist())
            label_list.extend(labels)
    result_collection = classification_report(label_list, pred_list, output_dict=True, target_names=REPORT_CLASS_NAMES, labels=REPORT_CLASS_LABELS)
    print(f"{desc} result:", result_collection, flush=True)
    return result_collection

def predict(model, dataloader):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            scores = model(data)
            labels = data["labels"]
            pred = torch.argmax(scores, dim=-1)
            max_label_length = data["max_label_length"]
            if max_label_length:
                n_doc = len(labels) // max_label_length
                assert len(labels) % max_label_length == 0
                for i in range(n_doc):
                    selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= 0
                    all_preds.append({
                        "doc_id": data["doc_id"][i],
                        "preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
                    })
    return all_preds

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=33637, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--log_steps", default=11212, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-7, type=float)
    parser.add_argument("--bert_lr", default=1e-7, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--sample_rate", default=0.1, type=float, help="randomly sample a portion of the training data")


    args = parser.parse_args()

    label_num = len(ID2REL)
    REPORT_CLASS_NAMES = [ID2REL[i] for i in range(1,len(ID2REL))]
    REPORT_CLASS_LABELS = list(range(1,len(ID2REL)))

    output_dir = Path(f"./output/{args.seed}/maven_ignore_none_{args.ignore_nonetype}_{args.sample_rate}")
    output_dir.mkdir(exist_ok=True, parents=True)
        
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizerFast.from_pretrained("/home/lichao/ERE/MAVEN-ERE/paramter/roberta/",add_prefix_space=True)

    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(tokenizer, "train", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype,sample_rate=args.sample_rate)
        dev_dataloader = get_dataloader(tokenizer, "valid",  max_length=256, shuffle=False, batch_size=args.batch_size * 2, ignore_nonetype=args.ignore_nonetype,sample_rate=args.sample_rate)
        train_doc_dataset =  myDocDataset(tokenizer, data_dir = "/home/lichao/ERE/MAVEN-ERE/data/MAVEN_ERE", split = 'train', max_length = 512, ignore_nonetype= None, sample_rate= 1)
        dev_doc_dataset =  myDocDataset(tokenizer, data_dir = "/home/lichao/ERE/MAVEN-ERE/data/MAVEN_ERE", split = 'valid', max_length = 512, ignore_nonetype= None, sample_rate= 1)

    ##test_dataloader = get_dataloader(tokenizer, "test", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype,sample_rate=args.sample_rate)
    

    print("loading model...")
    model = Model(len(tokenizer))
    model.half()
    model = to_cuda(model)

    if not args.eval_only:
        bert_optimizer = AdamW(model.parameters(), lr=args.bert_lr)
        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=2000, num_training_steps = len(train_dataloader) * args.epochs)
    eps = 1e-8

    Loss = nn.CrossEntropyLoss(ignore_index=-100)
    glb_step = 0
    train_doc = {}
    dev_doc = {}
    if not args.eval_only:
        print("*******************start training********************")
        train_losses = []
        pred_list = []
        label_list = []
        best_score = 0.0


        for i in range(len(train_doc_dataset)):
            doc = train_doc_dataset[i]
            train_doc[doc['doc_id']] = {
                'input_ids':doc['input_ids'],
                'event_spans':doc['event_spans'],
                'attention_mask':doc['attention_mask'], 
            }
        
        for i in range(len(dev_doc_dataset)):
            doc = dev_doc_dataset[i]
            dev_doc[doc['doc_id']] = {
                'input_ids':doc['input_ids'],
                'event_spans':doc['event_spans'],
                'attention_mask':doc['attention_mask'], 
            }
               
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):                
                model.train()
                loss = 0.0
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])

                output = model(data,train_doc)
                label = data['label']
                label_list.extend(label)
                label = torch.LongTensor(label)
                label = to_cuda(label)
                loss = Loss(output,label)
                
                pred = torch.argmax(output, dim=-1)
                pred = pred.cpu().numpy()
                pred_list.extend(pred)

                train_losses.append(loss.item())
                loss.backward()
                bert_optimizer.step()
                scheduler.step()
                bert_optimizer.zero_grad()

                glb_step += 1
                if glb_step % args.log_steps == 0 and glb_step != 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("true:")
                    print(label_list[:20])
                    print("pred:")
                    print(pred_list[:20])

                    print("Train %d steps: loss=%f" % (glb_step, np.mean(train_losses)))

                    res = classification_report(label_list, pred_list, output_dict=True, target_names=REPORT_CLASS_NAMES, labels=REPORT_CLASS_LABELS)
                    print("Train result:", res, flush = True)

                    train_losses = []
                    pred_list = []
                    label_list = []
                
                if glb_step % args.eval_steps == 0 and glb_step != 0:
                    res = evaluate(model, dev_dataloader, dev_doc, desc="Validation")

                    if "micro avg" not in res:
                        current_score = res["accuracy"]
                    else:
                        current_score = res["micro avg"]["f1-score"]
                    if current_score > best_score:
                        print("best result!", flush = True)
                        best_score = current_score
                        state = {"model":model.state_dict(), "optimizer":bert_optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                        torch.save(state, os.path.join(output_dir, "best"))

    
    print("*" * 30 + "Predict"+ "*" * 30)
    if os.path.exists(os.path.join(output_dir, "best")):
        print("loading from", os.path.join(output_dir, "best"))
        state = torch.load(os.path.join(output_dir, "best"))
        model.load_state_dict(state["model"])
    ##all_preds = predict(model, test_dataloader)
    ##temporal_dump("../data/MAVEN_ERE/test.jsonl", all_preds, output_dir, ignore_nonetype=args.ignore_nonetype)

    sys.stdout.close()