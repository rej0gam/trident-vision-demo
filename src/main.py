# TRIDENT AI Vision System
# Copyright (C) 2025  Rejey O. Gammad
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import argparse
import time
import math
from collections import defaultdict

import cv2
import threading
import queue
import numpy as np
import torch
import torch.nn.functional as F
import psutil
import GPUtil
import sys
from tqdm import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo

from torch.serialization import add_safe_globals
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from torchreid.reid.data.transforms import build_transforms
from torchreid.reid.utils import load_pretrained_weights
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchreid
from PIL import Image

add_safe_globals([np.core.multiarray.scalar])

# ------------------------------
# ASYNC
# ------------------------------
class FrameReader(threading.Thread):
    def __init__(self, src, queue_size=32):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                break
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            self.q.put(frame)
        self.cap.release()

    def read(self):
        return self.q.get()

    def more(self):
        return not self.q.empty() or not self.stopped

    def stop(self):
        self.stopped = True


class FrameWriter(threading.Thread):
    def __init__(self, path, fps, w, h, queue_size=32):
        super().__init__(daemon=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            try:
                frame = self.q.get(timeout=1)
                if frame is None:  # poison pill to stop gracefully
                    break
                self.out.write(frame)
            except queue.Empty:
                if self.stopped:
                    break

    def write(self, frame):
        self.q.put(frame)

    def stop(self):
        self.stopped = True
        self.out.release()

# ------------------------------
# Occlusion detector
# ------------------------------
class OcclusionHandler:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def get_occluded_indices(self, bboxes):
        """
        Returns a set of indices of detections considered occluded.
        bboxes: list of (x1,y1,x2,y2)
        """
        occluded = set()
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if iou(bboxes[i], bboxes[j]) > self.threshold:
                    occluded.add(i)
                    occluded.add(j)
        return occluded

# ------------------------------
# Detector
# ------------------------------
class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass

    def detect(self, frame, conf=0.45):
        """
        Returns list of detections: [{'bbox': [x1,y1,x2,y2], 'conf': float}]
        """
        results = self.model(frame, conf=conf, verbose=False, classes=[0])
        res = results[0]
        detections = []
        if hasattr(res, "boxes"):
            for det in res.boxes:
                xyxy = det.xyxy.cpu().numpy().flatten() if hasattr(det, "xyxy") else np.array(det.xyxy).flatten()
                conf_score = float(det.conf.cpu().numpy()) if hasattr(det, "conf") else float(det.conf)
                detections.append({
                    'bbox': xyxy.tolist(),
                    'conf': conf_score
                })
        return detections

# ------------------------------
# ReIdentifier
# ------------------------------
class ReIdentifier:
    def __init__(self, reid_weight_path, device=None, height=256, width=128, ema_alpha=0.9, matching_threshold=0.65):
        # Set device first
        self.device = device
        
        # Store other parameters
        self.reid_weight_path = reid_weight_path
        
        # Build model and load custom weights
        self.model = torchreid.models.build_model(
            name='osnet_ain_x1_0', 
            num_classes=0, 
            pretrained=False
        ).to(self.device)
        
        load_pretrained_weights(self.model, self.reid_weight_path)
        self.model.eval()
        
        # Rest of initialization remains the same
        self.transform, _ = build_transforms(
            height=height,
            width=width,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )
        self.stored_features = {}
        self.ema_alpha = ema_alpha
        self.matching_threshold = matching_threshold
        self.next_id = 1

    def extract_features_batch(self, crops):
        batch = []
        for img in crops:
            # Convert to torch tensor on GPU
            tensor = torch.from_numpy(img).to(self.device).permute(2,0,1).float() / 255.0
            tensor = TF.resize(tensor, [256, 128])  
            tensor = TF.normalize(tensor, [0.485,0.456,0.406], [0.229,0.224,0.225])
            batch.append(tensor)
        batch = torch.stack(batch).to(self.device)

        with torch.no_grad():
            feats = self.model(batch)
            feats = F.normalize(feats, dim=1)  # <-- now correct
        return feats

    def match_batch(self, features):
        """
        features: (N, D) CPU tensor
        returns: list of (assigned_id, similarity)
        """
        results = []

        if len(self.stored_features) == 0:
            # first entries
            for i in range(features.size(0)):
                new_id = self.next_id
                self.stored_features[new_id] = features[i:i+1].clone()
                results.append((new_id, 1.0))
                self.next_id += 1
            return results

        # stack stored features into a matrix (M, D)
        ids, stored_matrix = zip(*self.stored_features.items())
        stored_matrix = torch.cat(stored_matrix, dim=0)  # (M, D)

        # cosine similarity between all new feats and stored feats → (N, M)
        sims = torch.mm(features, stored_matrix.t())

        for i in range(features.size(0)):
            sim_vals = sims[i]  # (M,)
            best_sim, best_idx = torch.max(sim_vals, dim=0)
            best_id = ids[best_idx]

            if best_sim.item() >= self.matching_threshold:
                # EMA update
                old = self.stored_features[best_id]
                updated = self.ema_alpha * old + (1.0 - self.ema_alpha) * features[i:i+1]
                updated = F.normalize(updated, dim=1)
                self.stored_features[best_id] = updated
                results.append((best_id, best_sim.item()))
            else:
                # new ID
                new_id = self.next_id
                self.stored_features[new_id] = features[i:i+1].clone()
                results.append((new_id, 1.0))
                self.next_id += 1

        return results


# ------------------------------
# Monitoring helpers (psutil + GPUtil + pynvml)
# ------------------------------
def init_gpu_monitor():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        return handle
    except Exception as e:
        print("Warning: NVML init failed:", e)
        return None

def get_gpu_usage_nvml(handle):
    if handle is None:
        return {"gpu_util": 0.0, "gpu_mem_pct": 0.0}
    try:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)
        gpu_util = float(util.gpu)
        gpu_mem_pct = float(mem.used) / float(mem.total) * 100.0
        return {"gpu_util": gpu_util, "gpu_mem_pct": gpu_mem_pct}
    except Exception:
        return {"gpu_util": 0.0, "gpu_mem_pct": 0.0}

def get_system_usage(handle):
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu = get_gpu_usage_nvml(handle)
    return {"cpu": cpu, "ram": ram, "gpu_util": gpu["gpu_util"], "gpu_mem": gpu["gpu_mem_pct"]}

def print_progress(frames_measured, total_frames, device):
    pct = (frames_measured / total_frames) * 100
    mem = (torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 'cpu'
    sys.stdout.write(f"\rProgress: {pct:.1f}% | GPU Mem: {mem:.2f}GB")
    sys.stdout.flush()

def iou(boxA, boxB):
    # box = (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea + 1e-6)
    return interArea / union

# ------------------------------
# Main pipeline class
# ------------------------------
class VideoReIDPipeline:
    def __init__(self, 
                 det_model_path, 
                 reid_weight_path, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 nms_th=0.45, 
                 cosine_th=0.65, 
                 iou_th = 0.3, 
                 ema_alpha = 0.9,
                 use_ema = True,
                 use_overlay = True,
                 filter_class='all', 
                 box_color=(255, 100, 0), 
                 reid_batch_size=8):
        

        self.device = device
        self.detector = Detector(model_path=det_model_path, device=device)
        self.reidentifier = ReIdentifier(reid_weight_path, device=device, matching_threshold=cosine_th, ema_alpha=ema_alpha)
        self.reidentifier.model.eval()
        self.use_ema = use_ema
        self.use_overlay = use_overlay
        self.nms_th = nms_th
        self.cosine_th = cosine_th
        self.iou_th = iou_th
        self.ema_alpha = ema_alpha
        self.filter_class = filter_class
        self.box_color = box_color
        self.reid_batch_size = max(1, int(reid_batch_size))
        self.nvml_handle = init_gpu_monitor()
        self.occlusion_handler = OcclusionHandler(self.iou_th)
        self.stats = defaultdict(list)

    def clamp_bbox(self, bbox, w, h):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w - 1))
        y2 = max(0, min(int(round(y2)), h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def process_frame(self, frame):
        frame_h, frame_w = frame.shape[:2]

        # Detection
        t0 = time.time()
        detections = self.detector.detect(frame, conf=self.nms_th)
        t_det = time.time() - t0
        self.stats['det_time'].append(t_det)

        # Prepare crops
        crops, valid_dets = [], []
        for det in detections:
            bbox = det['bbox']
            conf = det.get('conf', 0.0)
            clamped = self.clamp_bbox(bbox, frame_w, frame_h)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_dets.append({'bbox': (x1, y1, x2, y2), 'conf': conf})

        # ReID
        assigned, t_reid_total = [], 0.0
        occluded_indices = set()
        if crops:
            occluded_indices = self.occlusion_handler.get_occluded_indices([d['bbox'] for d in valid_dets])
            for i in range(0, len(crops), self.reid_batch_size):
                batch_crops = crops[i:i + self.reid_batch_size]
                t0 = time.time()
                features = self.reidentifier.extract_features_batch(batch_crops)
                t1 = time.time()
                matches = self.reidentifier.match_batch(features)
                t2 = time.time()
                t_reid_total += (t1 - t0) + (t2 - t1)

                for k, (pid, sim) in enumerate(matches):
                    if (i + k) in occluded_indices:
                        matches[k] = (pid, sim)
                assigned.extend(matches)
        self.stats['reid_time'].append(t_reid_total)

        # Detections

        for idx, det in enumerate(valid_dets):
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            if idx < len(assigned):
                pid, sim = assigned[idx]
                label = f"ID:{pid} {sim:.2f}"
            else:
                label = "ID:-"
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), self.box_color, -1)
            cv2.putText(frame, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # System overlay
        if self.use_overlay:
            sysu = get_system_usage(self.nvml_handle)
            overlay_txt = f"CPU:{sysu['cpu']:.0f}% RAM:{sysu['ram']:.0f}% GPU:{sysu['gpu_util']:.0f}% GPU_MEM:{sysu['gpu_mem']:.0f}%"
            cv2.putText(frame, overlay_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 200, 20), 2)
        
        return frame, assigned, valid_dets, occluded_indices


    def process(self, input_video, output_video, show_progress=True):
        cap_tmp = cv2.VideoCapture(input_video)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap_tmp.release()

        reader = FrameReader(input_video)
        writer = FrameWriter(output_video, fps, w, h)
        reader.start()
        writer.start()

        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame", disable=not show_progress)
        frame_idx, t_start = 0, time.time()

        while reader.more():
            frame = reader.read()
            annotated = self.process_frame(frame)
            writer.write(annotated)

            frame_idx += 1
            pbar.update(1)
            print_progress(frame_idx, total_frames, self.device)

        t_total = time.time() - t_start
        pbar.close()
        
        writer.write(None)
        reader.stop()
        writer.stop()
        writer.join()
        reader.join()

        # Summary
        avg_det = np.mean(self.stats['det_time']) if self.stats['det_time'] else 0.0
        avg_reid = np.mean(self.stats['reid_time']) if self.stats['reid_time'] else 0.0
        avg_fps = frame_idx / t_total if t_total > 0 else 0.0

        print("\n--- Processing summary ---")
        print(f"Frames processed : {frame_idx}")
        print(f"Total time (s)   : {t_total:.2f}")
        print(f"Average FPS      : {avg_fps:.2f}")
        print(f"Avg detection (s): {avg_det:.4f}")
        print(f"Avg reid (s)     : {avg_reid:.4f}")
        print(f"Annotated video saved to: {output_video}")

# ------------------------------
# CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 + OSNet Video ReID pipeline")
    p.add_argument("--input", type=str, required=True, help="Path to input video")
    p.add_argument("--output", type=str, default="annotated.mp4", help="Path to output annotated video")
    p.add_argument("--det_model", type=str, default="src/data/models/yolov8n.pt", help="YOLOv8 model weights (default: yolov8s.pt)")
    p.add_argument("--reid_weights", type=str, default="src/data/models/osnet.pth.tar-10", help="Path to your fine-tuned OSNet weights")
    p.add_argument("--nms_th", type=float, default=0.45, help="Detection confidence threshold (default: 0.45)")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                   help="Device (cuda or cpu)")
    p.add_argument("--reid_batch", type=int, default=8, help="Batch size for ReID feature extraction")
    return p.parse_args()

def main():
    args = parse_args()
    pipeline = VideoReIDPipeline(
        det_model_path=args.det_model,
        reid_weight_path=args.reid_weights,
        device=args.device,
        nms_th=args.nms_th,
        box_color=(255, 100, 0),
        reid_batch_size=args.reid_batch
    )
    pipeline.process(args.input, args.output, show_progress=True)

if __name__ == "__main__":
    main()
