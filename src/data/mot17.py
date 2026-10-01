import os
import configparser
import csv
import numpy as np
import cv2
from typing import Dict, List, Tuple

def load_seqinfo(seq_path: str) -> dict:
    """
    Parse seqinfo.ini, return dict with metadata.
    """
    ini_path = os.path.join(seq_path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(ini_path)
    
    seq_info = {}
    if 'Sequence' in config:
        sec = config['Sequence']
        seq_info['name'] = sec.get('name', '')
        seq_info['imDir'] = sec.get('imDir', 'img1')
        seq_info['frameRate'] = int(sec.get('frameRate', 30))
        seq_info['seqLength'] = int(sec.get('seqLength', 0))
        seq_info['imWidth'] = int(sec.get('imWidth', 1920))
        seq_info['imHeight'] = int(sec.get('imHeight', 1080))
        seq_info['imExt'] = sec.get('imExt', '.jpg')
    return seq_info

def load_detections(seq_path: str) -> Dict[int, List[Dict]]:
    """
    Load det/det.txt
    Return {frame_id: [{'bbox': np.array([x1,y1,x2,y2]), 'confidence': float}, ...]}
    """
    det_path = os.path.join(seq_path, 'det', 'det.txt')
    detections: Dict[int, List[Dict]] = {}
    
    if not os.path.exists(det_path):
        return detections
        
    with open(det_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            frame_id = int(row[0])
            bb_left, bb_top, bb_width, bb_height = map(float, row[2:6])
            conf = float(row[6])
            
            x1, y1 = bb_left, bb_top
            x2 = bb_left + bb_width
            y2 = bb_top + bb_height
            
            if frame_id not in detections:
                detections[frame_id] = []
                
            detections[frame_id].append({
                'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                'confidence': conf
            })
            
    return detections

def load_ground_truth(seq_path: str) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[Dict]]]:
    """
    Load gt/gt.txt
    Filter: only class 1 (pedestrian), conf != 0 (which means it's considered for eval, or visibility)
    Return as tracks: {track_id: {frame_id: np.array([x1,y1,x2,y2])}}
    Also return as per-frame: {frame_id: [{'track_id': int, 'bbox': np.array, 'visibility': float}, ...]}
    """
    gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
    gt_tracks: Dict[int, Dict[int, np.ndarray]] = {}
    gt_per_frame: Dict[int, List[Dict]] = {}
    
    if not os.path.exists(gt_path):
        return gt_tracks, gt_per_frame
        
    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 9:
                continue
            frame_id = int(row[0])
            track_id = int(row[1])
            bb_left, bb_top, bb_width, bb_height = map(float, row[2:6])
            conf = float(row[6])
            cls_id = int(row[7])
            visibility = float(row[8])
            
            if cls_id != 1 or conf == 0:
                continue
                
            x1, y1 = bb_left, bb_top
            x2 = bb_left + bb_width
            y2 = bb_top + bb_height
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            
            if track_id not in gt_tracks:
                gt_tracks[track_id] = {}
            gt_tracks[track_id][frame_id] = bbox
            
            if frame_id not in gt_per_frame:
                gt_per_frame[frame_id] = []
            gt_per_frame[frame_id].append({
                'track_id': track_id,
                'bbox': bbox,
                'visibility': visibility
            })
            
    return gt_tracks, gt_per_frame

def list_sequences(mot17_root: str, split: str = 'train', det_type: str = 'SDP') -> List[str]:
    """
    List available sequence paths
    Filter by detector type (SDP/DPM/FRCNN)
    """
    split_dir = os.path.join(mot17_root, split)
    if not os.path.exists(split_dir):
        return []
    
    seqs = []
    for seq_name in os.listdir(split_dir):
        if seq_name.endswith(det_type):
            seqs.append(os.path.join(split_dir, seq_name))
            
    return sorted(seqs)

def get_train_val_split(det_type: str = 'SDP') -> Tuple[List[str], List[str]]:
    """
    Returns full sequence names with det_type suffix
    Train: MOT17-02, MOT17-04, MOT17-05, MOT17-11, MOT17-13
    Val: MOT17-09, MOT17-10
    """
    train_seqs = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-11', 'MOT17-13']
    val_seqs = ['MOT17-09', 'MOT17-10']
    
    train_split = [f"{s}-{det_type}" for s in train_seqs]
    val_split = [f"{s}-{det_type}" for s in val_seqs]
    
    return train_split, val_split

def load_frame_image(seq_path: str, frame_id: int) -> np.ndarray:
    """
    Load a specific frame image
    """
    seq_info = load_seqinfo(seq_path)
    im_dir = seq_info.get('imDir', 'img1')
    ext = seq_info.get('imExt', '.jpg')
    img_name = f"{frame_id:06d}{ext}"
    img_path = os.path.join(seq_path, im_dir, img_name)
    
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
