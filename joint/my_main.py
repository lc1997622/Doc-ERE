import sys
sys.path.append("../")
sys.path.append("src/")
from src.utils import to_cuda
from src.test_model import Model
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data import get_dataloader
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from src.data import REL2ID, ID2REL
import warnings
import os
import sys
from pathlib import Path 
from src.dump_result import temporal_dump

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

def evaluate(model, dataloader, desc=""):
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
            scores = model(data)
            labels = data["labels"]
            scores = scores.view(-1, scores.size(-1))
            labels = labels.view(-1)
            pred = torch.argmax(scores, dim=-1)
            pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
            label_list.extend(labels[labels>=0].cpu().numpy().tolist())
    result_collection = classification_report(label_list, pred_list, output_dict=True, target_names=REPORT_CLASS_NAMES, labels=REPORT_CLASS_LABELS)
    print(f"{desc} result:", result_collection)
    return result_collection

def predict(model, dataloader):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predict"):
            pred_list = []
            label_list = []
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            sent_output = model(data)

            labels = data['pre_rlabel']
            for label in labels:
                label_str = tokenizer.decode(label)
                label_list.append(label_str)
            labels = torch.unsqueeze(labels, -1)
            zero_l =  to_cuda(torch.zeros(labels.shape[0],labels.shape[1],len(tokenizer)))
            labels = zero_l.scatter(2,labels,1)
            pred = torch.argmax(sent_output, dim=-1)
            
            pred = pred.cpu().numpy()

            for pre in pred:
                pre_str = tokenizer.decode(pre)
                pred_list.append(pre_str)
            n_doc = len(data['pre_rlabel'])
            for i in range(n_doc):
                all_preds.append({
                    "doc_id": data["doc_id"][i],
                    "preds": pred_list,
                })
    return all_preds



if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--log_steps", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--sample_rate", default=1, type=float, help="randomly sample a portion of the training data")


    args = parser.parse_args()

    label_num = len(ID2REL)
    REPORT_CLASS_NAMES = [ID2REL[i] for i in range(0,len(ID2REL) - 1)]
    REPORT_CLASS_LABELS = list(range(len(ID2REL) - 1))

    output_dir = Path(f"./output/{args.seed}/maven_ignore_none_{args.ignore_nonetype}_{args.sample_rate}")
    output_dir.mkdir(exist_ok=True, parents=True)
        
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("loading data...")
    if not args.eval_only:
        train_dataloader = get_dataloader(tokenizer, "train", max_length=256, shuffle=True, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
        dev_dataloader = get_dataloader(tokenizer, "valid",  max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
    test_dataloader = get_dataloader(tokenizer, "test", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)

    print("loading model...")
    model = Model(len(tokenizer))
    model = to_cuda(model)

    if not args.eval_only:
        bert_optimizer = AdamW(model.parameters(), lr=args.bert_lr)
        scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * args.epochs)
    eps = 1e-8

    Loss = nn.CrossEntropyLoss(ignore_index=-100)
    glb_step = 0
    if not args.eval_only:
        print("*******************start training********************")
        train_losses = []
        best_score = 0.0
        for epoch in range(args.epochs):
            for data in tqdm(train_dataloader, desc=f"Training epoch {epoch}"):
                model.train()
                loss = 0.0
                pred_list = []
                label_list = []
                for k in data:
                    if isinstance(data[k], torch.Tensor):
                        data[k] = to_cuda(data[k])
                sent_output = model(data)


                labels = data['pre_rlabel']
                for label in labels:
                    label_str = tokenizer.decode([l for l in label if l != -100])
                    label_list.append(label_str)
                
                labels = to_cuda(labels)
                loss = Loss(sent_output.view(-1, len(tokenizer)), labels.view(-1))

                pred = torch.argmax(sent_output, dim=-1)
                pred = pred.cpu().numpy()

                for pre in pred:
                    pre_str = tokenizer.decode(pre)
                    pred_list.append(pre_str)

                train_losses.append(loss.item())
                loss.backward()
                bert_optimizer.step()
                scheduler.step()
                bert_optimizer.zero_grad()

                glb_step += 1

                if epoch % 20 == 0:
                    print("e")

                if glb_step % args.log_steps == 0:
                    print("*"*20 + "Train Prediction Examples" + "*"*20)
                    print("true:")
                    print(label_list[:10])
                    print("pred:")
                    print(pred_list[:10])
                    print("Train %d steps: loss=%f" % (glb_step, np.mean(train_losses)))
                    
                    train_losses = []
                    pred_list = []
                    label_list = []
                

                if glb_step % args.eval_steps == 0:
                    0==0
                    # res = evaluate(model, dev_dataloader, desc="Validation")

                    # if "micro avg" not in res:
                    #     current_score = res["accuracy"]
                    # else:
                    #     current_score = res["micro avg"]["f1-score"]
                    # if current_score > best_score:
                    #     print("best result!")
                    #     best_score = current_score
                    #     state = {"model":model.state_dict(), "optimizer":bert_optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                    #     torch.save(state, os.path.join(output_dir, "best"))

    
    print("*" * 30 + "Predict"+ "*" * 30)
    if os.path.exists(os.path.join(output_dir, "best")):
        print("loading from", os.path.join(output_dir, "best"))
        state = torch.load(os.path.join(output_dir, "best"))
        model.load_state_dict(state["model"])
    all_preds = predict(model, test_dataloader)
    ##temporal_dump("../data/MAVEN_ERE/test.jsonl", all_preds, output_dir, ignore_nonetype=args.ignore_nonetype)

    sys.stdout.close()