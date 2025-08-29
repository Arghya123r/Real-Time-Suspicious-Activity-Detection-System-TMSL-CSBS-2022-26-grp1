#!/usr/bin/env python3
import os
import csv
import random
import glob
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Optional, List

# --------------- Config ---------------
class Cfg:
    dataset_root = "/kaggle/input/training-processed/data/processed"#<-- changes based on database location
    train_manifest_csv = "kaggle_train_manifest.csv"
    val_manifest_csv = "kaggle_val_manifest.csv"
    # Sequence parameters
    frames_per_sequence = 16
    num_sequences = 8
    sequence_overlap = 8
    # Image processing
    frame_size = 224
    # Model parameters
    backbone = "resnet18"
    pretrained = True
    freeze_cnn = False
    lstm_hidden = 128
    lstm_layers = 1
    bidirectional = False
    dropout = 0.3
    # Training parameters
    batch_size = 32
    num_workers = 4
    prefetch_factor = 2
    pin_memory = True
    epochs = 5
    lr = 1e-2
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MIL parameters
    use_mil = True
    mil_margin = 1.0
    topk = 1
    clip_grad = 5.0
    # Data split
    train_split = 0.8
    # AMP & scheduler
    max_lr = 1e-2
cfg = Cfg()
print("Using: ", cfg.device)

# --------------- Transforms ---------------
img_transform = transforms.Compose([
    transforms.Resize((cfg.frame_size, cfg.frame_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------- Data Utilities ---------------
def create_kaggle_manifests():
    anomaly_classes = {
        'Abuse','Arrest','Arson','Assault','Burglary','Explosion',
        'Fighting','RoadAccidents','Robbery','Shooting','Shoplifting',
        'Stealing','Vandalism'
    }
    normal_classes = {'Normal','Normal_Videos','normal'}
    class_dirs = [d for d in os.listdir(cfg.dataset_root)
                  if os.path.isdir(os.path.join(cfg.dataset_root, d))]
    train_data, val_data = [], []
    print("Processing Kaggle UCF-Crime dataset...")
    for class_name in class_dirs:
        class_path = os.path.join(cfg.dataset_root, class_name)
        if class_name in anomaly_classes:
            label = 1
        elif any(n.lower() in class_name.lower() for n in normal_classes):
            label = 0
        else:
            label = 1
            print(f"Warning: Unknown class '{class_name}' assumed anomalous")
        all_images = []
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
            all_images.extend(glob.glob(os.path.join(class_path, ext)))
        all_images.sort()
        # group by video id
        video_groups = defaultdict(list)
        for p in all_images:
            name = os.path.basename(p)
            vid = '_'.join(name.split('_')[:-1]) if '_' in name else name.split('.')[0]
            video_groups[vid].append(p)
        seqs = []
        step = cfg.frames_per_sequence - cfg.sequence_overlap
        for frames in video_groups.values():
            frames.sort()
            for i in range(0, len(frames)-cfg.frames_per_sequence+1, step):
                seqs.append(frames[i:i+cfg.frames_per_sequence])
        if not seqs and len(all_images) >= cfg.frames_per_sequence:
            for i in range(0, len(all_images)-cfg.frames_per_sequence+1, step):
                seqs.append(all_images[i:i+cfg.frames_per_sequence])
        random.shuffle(seqs)
        split = int(len(seqs)*cfg.train_split)
        for s in seqs[:split]:
            train_data.append([';'.join(s), label])
        for s in seqs[split:]:
            val_data.append([';'.join(s), label])
        print(f"  {class_name}: {len(all_images)} images -> {split} train, {len(seqs)-split} val")
    for fname, data in [(cfg.train_manifest_csv, train_data),
                        (cfg.val_manifest_csv, val_data)]:
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame_paths','label'])
            w.writerows(data)
    print(f"Manifests created: {len(train_data)} train, {len(val_data)} val")
    return len(train_data), len(val_data)

class KaggleFrameDataset(Dataset):
    def __init__(self, manifest_csv):
        self.items = []
        with open(manifest_csv, newline='') as f:
            for row in csv.DictReader(f):
                paths = row['frame_paths'].split(';')
                self.items.append((paths, int(row['label'])))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        paths, label = self.items[idx]
        frames = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                frames.append(img_transform(img))
            except:
                frames.append(torch.zeros(3, cfg.frame_size, cfg.frame_size))
        bag = torch.stack(frames).unsqueeze(0)  # (1, F, C, H, W)
        return bag, torch.tensor(label, dtype=torch.long)

def collate_frame_bags(batch):
    bags, labels = zip(*batch)
    return torch.stack(bags), torch.stack(labels)

from typing import Optional, List

def format_lrs(optimizer, scheduler: Optional[object] = None, mode: str = "all") -> str:
    """
    Build a display string for current LR(s).
    mode: "first" -> first group's lr; "mean" -> mean across groups; "all" -> join all.
    """
    # Collect current LRs as a list of floats
    if scheduler is not None and hasattr(scheduler, "get_last_lr"):
        lrs: List[float] = list(scheduler.get_last_lr())
    else:
        lrs = [pg["lr"] for pg in optimizer.param_groups]

    if not lrs:  # safety
        return "n/a"

    if mode == "first" or len(lrs) == 1:
        return f"{lrs[0]:.2e}"  # Access first element, not the entire list
    elif mode == "mean":
        return f"{(sum(lrs) / len(lrs)):.2e}"
    elif mode == "all":
        return " / ".join(f"{v:.2e}" for v in lrs)
    else:
        return f"{lrs[0]:.2e}"  # Fallback to first element
# --------------- Model ---------------
class LRCN(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True,
                 lstm_hidden=256, lstm_layers=1, bidirectional=False,
                 dropout=0.3, freeze_cnn=False):
        super().__init__()
        if cfg.backbone=='resnet18':
            net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if cfg.pretrained else None)
            feat_dim=960
            print("Using Mobilenet")
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if cfg.pretrained else None)
            feat_dim=2048
        self.cnn = nn.Sequential(*list(net.children())[:-1])
        net.to(cfg.device)
        if cfg.freeze_cnn:
            for p in self.cnn.parameters(): p.requires_grad=False
            print("CNN Frozen")
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=cfg.lstm_hidden,
                            num_layers=cfg.lstm_layers, batch_first=True,
                            bidirectional=cfg.bidirectional,
                            dropout=0.0 if cfg.lstm_layers==1 else cfg.dropout)
        out_dim = cfg.lstm_hidden*(2 if cfg.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim,128), nn.ReLU(inplace=True),
            nn.Linear(128,1)
        )
    #def forward(self, bag):
    #    B,S,F,C,H,W = bag.shape
    #    x = bag.view(B*S*F,C,H,W)
    #    feats = self.cnn(x).view(B,S,F,-1)
    #    logits=[]
    #    for s in range(S):
    #        seq = feats[:,s]
    #        lstm_out,_=self.lstm(seq)
    #        pooled=lstm_out[:,-1]
    #        logits.append(self.classifier(pooled).squeeze(-1))
    #    return torch.stack(logits,1)
    def forward(self,bag):
        B,S,F,C,H,W = bag.shape
        x = bag.view(B*S*F,C,H,W)
        feats = self.cnn(x)
        
        feat_dim = feats.shape[1]
        feats = feats.view(B,S,F,feat_dim)
        
        lstm_out, _ = self.lstm(feats)
        pooled = lstm_out[:,-1,:]
        
        logits = self.classifier(pooled)
        
        return logits
        
        
        

def aggregate_video_score(seg_logits, mode="max", k=1):
    if mode == "max":
        v, _ = seg_logits.max(dim=1)
    elif mode == "mean":
        v = seg_logits.mean(dim=1)
    elif mode == "topk":
        k = max(1, min(k, seg_logits.shape[1]))
        v, _ = torch.topk(seg_logits, k=k, dim=1)
        v = v.mean(dim=1)
    return v

def mil_ranking_loss(pos_seg, neg_seg, margin=1.0, topk=1):
    k = max(1, min(topk, pos_seg.shape[1], neg_seg.shape[1]))
    pos_top, _ = torch.topk(pos_seg, k=k, dim=1)
    neg_top, _ = torch.topk(neg_seg, k=k, dim=1)
    pos_score = pos_top.mean(dim=1).unsqueeze(1)
    neg_score = neg_top.mean(dim=1).unsqueeze(0)
    loss = F.relu(margin - pos_score + neg_score).mean()
    return loss

def train_epoch(model, loader, optimizer, scaler, scheduler=None, epoch: int = 0):
    model.train()
    running = 0.0
    bce = nn.BCEWithLogitsLoss()

    pbar = tqdm(enumerate(loader),
                total=len(loader),
                desc=f"Epoch {epoch}",
                dynamic_ncols=True)

    for i, (bags, labels) in pbar:
        bags, labels = bags.to(cfg.device), labels.float().to(cfg.device)

        with autocast(cfg.device):
            optimizer.zero_grad()
            seg_logits = model(bags)
            if cfg.use_mil:
                pos = labels == 1
                neg = labels == 0
                loss = torch.tensor(0.0, device=cfg.device)
                if pos.any() and neg.any():
                    loss += mil_ranking_loss(seg_logits[pos], seg_logits[neg], cfg.mil_margin, cfg.topk)
                vids = aggregate_video_score(seg_logits, mode="topk", k=cfg.topk)
                loss += bce(vids, labels)
            else:
                vids = aggregate_video_score(seg_logits, mode="topk", k=cfg.topk)
                loss = bce(vids, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        running += loss.item() * bags.size(0)

        # update tqdm postfix with current/avg loss and LR
        lr_str = format_lrs(optimizer, scheduler, mode = "mean")
        avg_loss = running / ((i + 1) * bags.size(0))
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=lr_str)

    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_s,all_l=[],[]
    for bags,labels in loader:
        bags=bags.to(cfg.device)
        seg_logits=model(bags)
        vids=torch.sigmoid(aggregate_video_score(seg_logits,"topk",cfg.topk))
        all_s.append(vids.cpu()); all_l.append(labels)
    scores=torch.cat(all_s).numpy(); labels=torch.cat(all_l).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return {"AUC":roc_auc_score(labels,scores)}
    except:
        preds=(scores>=0.5).astype(int)
        return {"ACC":(preds==labels).mean()}

def main():
    if not (os.path.exists(cfg.train_manifest_csv) and os.path.exists(cfg.val_manifest_csv)):
        create_kaggle_manifests()

    train_ds=KaggleFrameDataset(cfg.train_manifest_csv)
    val_ds=KaggleFrameDataset(cfg.val_manifest_csv)
    train_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True,
                            num_workers=cfg.num_workers,pin_memory=cfg.pin_memory,
                            prefetch_factor=cfg.prefetch_factor,collate_fn=collate_frame_bags)
    val_loader=DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,
                          num_workers=cfg.num_workers,pin_memory=cfg.pin_memory,
                          prefetch_factor=cfg.prefetch_factor,collate_fn=collate_frame_bags)

    model=LRCN(backbone=cfg.backbone, pretrained=cfg.pretrained,
                lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
                bidirectional=cfg.bidirectional, dropout=cfg.dropout,
                freeze_cnn=cfg.freeze_cnn)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model,device_ids = [0,1])
    model.to(cfg.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scaler=GradScaler()
    scheduler=OneCycleLR(optimizer,max_lr=cfg.max_lr,
                        steps_per_epoch=len(train_loader),epochs=cfg.epochs)

    best=-1
    for epoch in range(1,cfg.epochs+1):
        print("Training Epoch Started.")
        loss=train_epoch(model,train_loader,optimizer,scaler,scheduler,epoch)
        mets=evaluate(model,val_loader)
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | " +
              " ".join(f"{k}:{v:.4f}" for k,v in mets.items()))
        score=mets.get("AUC",mets.get("ACC",0))
        if score>best:
            best=score
            torch.save(model.state_dict(),"lrcn_ucf_best.pt")
            print(f"  -> New best: {best:.4f}")

if __name__=="__main__":
    main()
