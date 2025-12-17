# %% [code] {"execution":{"iopub.status.busy":"2025-12-16T13:57:36.685387Z","iopub.execute_input":"2025-12-16T13:57:36.685564Z","iopub.status.idle":"2025-12-16T13:57:47.925259Z","shell.execute_reply.started":"2025-12-16T13:57:36.685547Z","shell.execute_reply":"2025-12-16T13:57:47.924725Z"}}
#!/usr/bin/env python3
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from collections import deque
import argparse

# %% [code] {"execution":{"iopub.status.busy":"2025-12-16T13:57:47.926628Z","iopub.execute_input":"2025-12-16T13:57:47.926903Z","iopub.status.idle":"2025-12-16T13:57:48.018745Z","shell.execute_reply.started":"2025-12-16T13:57:47.926886Z","shell.execute_reply":"2025-12-16T13:57:48.018149Z"}}
class Cfg:
    # Sequence parameters
    frames_per_sequence = 10
    num_sequences = 6
    sequence_overlap = 6
    # Image processing
    frame_size = 224
    # Model parameters
    backbone = "resnet18"
    pretrained = True
    freeze_cnn = True
    lstm_hidden = 256
    lstm_layers = 2
    bidirectional = True
    dropout = 0.2
    # Training parameters
    batch_size = 16
    num_workers = 4
    prefetch_factor = 2
    pin_memory = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topk = 3
cfg = Cfg()
print("Using: ", cfg.device)

# %% [code] {"execution":{"iopub.status.busy":"2025-12-16T13:57:48.019393Z","iopub.execute_input":"2025-12-16T13:57:48.019582Z","iopub.status.idle":"2025-12-16T13:57:48.042605Z","shell.execute_reply.started":"2025-12-16T13:57:48.019567Z","shell.execute_reply":"2025-12-16T13:57:48.042076Z"}}
# --------------- Model Definition (Same as Training) ---------------
class LRCN(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True,
                 lstm_hidden=256, lstm_layers=1, bidirectional=False,
                 dropout=0.3, freeze_cnn=False):
        super().__init__()
        if cfg.backbone=='resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None)
            feat_dim=512
            print("Using resnet18")
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
        feats = feats.view(B,S*F,feat_dim)
        
        lstm_out, _ = self.lstm(feats)
        pooled = lstm_out[:,-1,:]
        
        logits = self.classifier(pooled)
        
        return logits
        

# %% [code] {"execution":{"iopub.status.busy":"2025-12-16T13:57:48.043284Z","iopub.execute_input":"2025-12-16T13:57:48.043555Z","iopub.status.idle":"2025-12-16T13:57:48.067536Z","shell.execute_reply.started":"2025-12-16T13:57:48.043529Z","shell.execute_reply":"2025-12-16T13:57:48.066940Z"}}
# --------------- Video Processing Class ---------------
class SuspiciousActivityDetector:
    def __init__(self, model_path, device='cuda', threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.frames_per_sequence = cfg.frames_per_sequence
        self.frame_size = cfg.frame_size
        
        # Load model
        self.model = LRCN()
        checkpoint = torch.load(model_path, map_location=self.device)
        # Handle DataParallel wrapper if present
        if 'module.' in list(checkpoint.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            checkpoint = new_state_dict
        
        self.model.load_state_dict(checkpoint)
        if torch.cuda.device_count()>1:
            self.model=nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")

    def extract_and_predict(self, video_path, output_path, display_live=False):
        """Extract frames from video and predict suspicious activities"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Frame buffer for sequences
        frame_buffer = deque(maxlen=2*self.frames_per_sequence)
        prediction_buffer = deque(maxlen=self.frames_per_sequence)
        
        frame_count = 0
        PROCESS_EVERY_N_FRAMES = 2 

        all_predictions = [] 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame for model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            tensor_frame = self.transform(pil_frame)
            frame_buffer.append(tensor_frame)
            
            # Predict when buffer is full
            if len(frame_buffer) == 2*self.frames_per_sequence:
                if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                    prediction = self._predict_sequence(frame_buffer)
                prediction_buffer.append(prediction)
            else:
                # Use previous prediction or default for early frames
                prediction = prediction_buffer[-1] if prediction_buffer else 0.0
                prediction_buffer.append(prediction)

            all_predictions.append(prediction_buffer[-1])
            
            # Add overlay to frame
            overlay_frame = self._add_overlay(frame, prediction_buffer[-1])
            
            # Write frame to output video
            out.write(overlay_frame)
            
            # Display live preview if requested
            #if display_live:
            #    cv2.imshow('Suspicious Activity Detection', overlay_frame)
            #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        #if display_live:
        #    cv2.destroyAllWindows()
        
        print(f"Output video saved to: {output_path}")
        return self._generate_summary(all_predictions)
        
    def aggregate_video_score(self,seg_logits, mode="mean", k=1):
        if mode == "max":
            v, _ = seg_logits.max(dim=1)
        elif mode == "mean":
            v = seg_logits.mean(dim=1)
        elif mode == "topk":
            k = max(1, min(k, seg_logits.shape[1]))
            v, _ = torch.topk(seg_logits, k=k, dim=1)
            v = v.mean(dim=1)
        return v

    def _predict_sequence(self, frame_buffer):
        """Predict suspiciousness for current frame sequence"""
        with torch.no_grad():
            # Stack frames and create bag: (1, 1, F, C, H, W)
            sequence = torch.stack(list(frame_buffer)).unsqueeze(0).unsqueeze(0)
            sequence = sequence.to(self.device)
            
            # Get prediction
            seg_logits = self.model(sequence)
            vids=torch.sigmoid(self.aggregate_video_score(seg_logits,mode="topk",k=cfg.topk))
            prediction = vids.cpu().item()
        return prediction

    #def _predict_sequence(self, frame_buffer):
    #    """Predict suspiciousness for current frame sequence"""
    #    with torch.no_grad():
    #        # Stack frames and create bag: (1, 1, F, C, H, W)
    #        sequence = torch.stack(list(frame_buffer)).unsqueeze(0).unsqueeze(0)
    #        sequence = sequence.to(self.device)
    #        
    #        # Get prediction
    #        seg_logits = self.model(sequence)
    #        vids=torch.sigmoid(seg_logits)
    #        prediction = vids.cpu().item()
    #    return prediction

    def _add_overlay(self, frame, prediction_score):
        """Add prediction overlay to frame"""
        overlay_frame = frame.copy()
        
        # Determine status and color
        is_suspicious = prediction_score >= self.threshold
        status = "SUSPICIOUS" if is_suspicious else "NORMAL"
        color = (0, 0, 255) if is_suspicious else (0, 255, 0)  # Red for suspicious, Green for normal
        
        # Add semi-transparent background for text
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, overlay_frame, 0.7, 0, overlay_frame)
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay_frame, f"Status: {status}", (20, 40), font, 0.8, color, 2)
        cv2.putText(overlay_frame, f"Confidence: {prediction_score:.3f}", (20, 70), font, 0.7, (255, 255, 255), 2)
        
        # Add progress bar for confidence
        bar_width = 200
        bar_height = 10
        bar_x, bar_y = 20, 85
        
        # Background bar
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Confidence bar
        conf_width = int(bar_width * prediction_score)
        cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1)
        
        # Add timestamp
        timestamp = f"Frame: {cv2.getTickCount()}"
        cv2.putText(overlay_frame, timestamp, (overlay_frame.shape[1] - 200, 30), font, 0.5, (255, 255, 255), 1)
        
        return overlay_frame

    def _generate_summary(self, predictions):
        """Generate summary statistics"""
        predictions_list = list(predictions)
        if not predictions_list:
            return {}
        
        suspicious_frames = sum(1 for p in predictions_list if p >= self.threshold)
        total_frames = len(predictions_list)
        max_confidence = max(predictions_list)
        avg_confidence = sum(predictions_list) / total_frames
        
        summary = {
            'total_frames': total_frames,
            'suspicious_frames': suspicious_frames,
            'suspicious_percentage': (suspicious_frames / total_frames) * 100,
            'max_confidence': max_confidence,
            'avg_confidence': avg_confidence
        }
        
        return summary

# %% [markdown] {"execution":{"iopub.status.busy":"2025-08-28T07:47:49.253552Z","iopub.execute_input":"2025-08-28T07:47:49.253808Z","iopub.status.idle":"2025-08-28T07:47:49.345783Z","shell.execute_reply.started":"2025-08-28T07:47:49.253781Z","shell.execute_reply":"2025-08-28T07:47:49.344879Z"}}
# # --------------- Real-time Processing (Optional) ---------------
# class RealTimeDetector(SuspiciousActivityDetector):
#     def __init__(self, model_path, device='cuda', threshold=0.5):
#         super().__init__(model_path, device, threshold)
#     
#     def process_webcam(self, camera_id=0):
#         """Process live webcam feed"""
#         cap = cv2.VideoCapture(camera_id)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open camera {camera_id}")
#         
#         frame_buffer = deque(maxlen=self.frames_per_sequence)
#         
#         print("Starting real-time detection. Press 'q' to quit.")
#         
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             
#             # Preprocess frame
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_frame = Image.fromarray(rgb_frame)
#             tensor_frame = self.transform(pil_frame)
#             frame_buffer.append(tensor_frame)
#             
#             # Predict if buffer is full
#             if len(frame_buffer) == self.frames_per_sequence:
#                 prediction = self._predict_sequence(frame_buffer)
#             else:
#                 prediction = 0.0  # Default for initial frames
#             
#             # Add overlay and display
#             overlay_frame = self._add_overlay(frame, prediction)
#             cv2.imshow('Real-time Suspicious Activity Detection', overlay_frame)
#             
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         
#         cap.release()
#         cv2.destroyAllWindows()

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2025-12-16T14:02:40.259171Z","iopub.execute_input":"2025-12-16T14:02:40.259785Z","iopub.status.idle":"2025-12-16T14:03:05.371018Z","shell.execute_reply.started":"2025-12-16T14:02:40.259757Z","shell.execute_reply":"2025-12-16T14:03:05.370252Z"}}
# --------------- Main Function ---------------
def main():
    parser = argparse.ArgumentParser(description='Suspicious Activity Detection in Videos')
    #parser.add_argument('--model', required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--input', help='Input video path')
    #parser.add_argument('--output', help='Output video path')
    #parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    #parser.add_argument('--camera_id', type=int, default=0, help='Camera ID for webcam')
    #parser.add_argument('--threshold', type=float, default=0.5, help='Suspicion threshold')
    #parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    #parser.add_argument('--display', action='store_true', help='Display live preview')
    
    args = parser.parse_args()
    
    #if args.webcam:
    #    # Real-time webcam processing
    #    detector = RealTimeDetector(args.model, args.device, args.threshold)
    #   detector.process_webcam(args.camera_id)
    model = 'Model\lrcn_ucf_best.pt'
    output_dir = 'output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.5
    input_1 = args.input
    filename = os.path.basename(input_1)
    output_filename = 'output_' + filename
        # Video file processing
    output = os.path.join(output_dir,output_filename)
    detector = SuspiciousActivityDetector(model, device, threshold)
    
    display=False
    summary = detector.extract_and_predict(input_1, output, display)
        
        # Print summary
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    print(f"Total frames processed: {summary['total_frames']}")
    print(f"Suspicious frames: {summary['suspicious_frames']}")
    print(f"Suspicious percentage: {summary['suspicious_percentage']:.2f}%")
    print(f"Maximum confidence: {summary['max_confidence']:.3f}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()

# Example usage:
# python suspicious_detection.py --model lrcn_ucf_best.pt --input input_video.mp4 --output output_video.mp4 --display
# python suspicious_detection.py --model lrcn_ucf_best.pt --webcam --camera_id 0

# %% [code]
