import os, cv2, torch, numpy as np
from torchvision import models, transforms
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import math
from torch.nn.functional import cosine_similarity


# ---------- Config ----------
FRAME_DIR        = './rc-cars/rc-cars'         # video frames directory
INIT_CENTER_PT   = (566, 246)                # starting coordinates, also support multiple init center point, e.g., [(566, 246), (359, 147)]
FRAME_SIZE       = (640, 360)                # pixels resolution
OBJECT_DETECTOR  = 'fasterrcnn_resnet50_fpn'
OUTPUT_VIDEO     = "tracked_rc_car_no_app_no_kl.mp4"   # output video file
FPS              = 30                        # frame rate
OUTPUT_FRAME_DIR = './output_frame_no_app_no_kl'
os.makedirs(OUTPUT_FRAME_DIR, exist_ok=True)

IOU_THR          = 0.25                      # accept the minimum IoU for matching
TARGET_CLASS_ID  = 3                         # the ID of car in COCO is 3
MAX_MISS        = 10
MAX_LOST_AGE    = 30
HIST_LEN        = 5                          # the length of history bbox                        

# Cost weight
W_IOU           = 0.7
W_DIST          = 0.3


# Tracking data structure ----------
class Track:
    def __init__(self, box, tid):
        self.box = box         # the bbox (xyxy) of object in previous frame
        self.tid = tid         # the tracting object ID
        self.miss_frames = 0           # the number of frames that the object ID is missing

        self.box_history = []
        self.lost_age = 0

    def update(self, new_box):
        self.box_history.append(self.box)
        self.box = new_box
        if len(self.box_history) > HIST_LEN:
            self.box_history.pop(0)

    def check_consistency(self, new_box):
        if len(self.box_history) > 0:
            prev_box = self.box_history[-1] 
            iou = iou_xyxy(prev_box, new_box)
            dist = center_dist(prev_box, new_box)
            cost = W_IOU * (1 - iou) +  W_DIST * dist
            if cost < (1 - IOU_THR):
                return True
            else: 
                return False
        else:
            return True


def load_detector(model_name: str, device):
    """
    Load an object detector from torchvision based on the model name.

    Args:
        model_name (str): Name of the object detector supported by torchvision.models.detection

    Returns:
        model (torch.nn.Module): Loaded detector model in eval mode and on the appropriate device.
    """
    supported_models = {
        'fasterrcnn_resnet50_fpn': models.detection.fasterrcnn_resnet50_fpn,
        'ssd300_vgg16': models.detection.ssd300_vgg16,
    }

    if model_name not in supported_models:
        raise ValueError(f"Unsupported detector model: {model_name}. Supported models are: {list(supported_models.keys())}")

    model = supported_models[model_name](pretrained=True)
    model.to(device)
    model.eval()

    return model


def iou_xyxy(boxA, boxB):
    """IoU of two boxes, xyxy format"""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def center_dist(boxA, boxB):
    ca = np.array([(boxA[0]+boxA[2])/2, (boxA[1]+boxA[3])/2])
    cb = np.array([(boxB[0]+boxB[2])/2, (boxB[1]+boxB[3])/2])
    max_dist = math.sqrt(FRAME_SIZE[0]**2 + FRAME_SIZE[1]**2)
    return min(1.0, np.linalg.norm(ca - cb) / max_dist)  # normalize


def detect(model, frame):
    """Run detector, filter by class & score"""
    with torch.no_grad():
        out = model([frame])[0]
    return out['boxes'].cpu().numpy(), out['scores'].cpu().numpy()


def init_object_trajectory(first_frame, init_center_points, model, device):
    """
    Given the initial center points, we identify the corresponding initial bounding box for each point
    """
    if not isinstance(init_center_points[0], tuple):
        init_center_points = [init_center_points]

    init_frame = cv2.imread(first_frame)
    transform = transforms.Compose([transforms.ToTensor()])
    frame_tensor = transform(init_frame).to(device)
    dets, _  = detect(model, frame_tensor)

    tracks = []                 #
    lost_tracks  = []
    next_id = 0   
    for center_point in init_center_points:
        if len(dets):
            dists = [np.linalg.norm([(d[0]+d[2])/2 - center_point[0],
                                    (d[1]+d[3])/2 - center_point[1]]) for d in dets]
            init_box = dets[int(np.argmin(dists))]
        else:
            # If the object detector fails to detect any object in the first frame, we initialize the track with a minimal bounding box tightly enclosing the given center point.
            x, y = center_point
            init_box = np.array([x-10, y-10, x+10, y+10])
        tracks.append(Track(init_box, next_id)); next_id += 1
    return tracks, lost_tracks


def video_tractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    print(f'Loading object detector from {OBJECT_DETECTOR}...')
    model = load_detector(model_name=OBJECT_DETECTOR, device=device)

    # obtaining all video frames
    frames = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg'))
    frames = [os.path.join(FRAME_DIR, f) for f in frames]

    # video writor
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, FRAME_SIZE)

    tracks, lost_tracks = init_object_trajectory(frames[0], INIT_CENTER_PT, model, device)

    frame_idx = 0

    for fpath in tqdm(frames, desc="Tracking"):
        frame = cv2.imread(fpath)
        transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = transform(frame).to(device)
        det_boxes, det_scores = detect(model, frame_tensor)
            
        n_det = len(det_boxes)
        n_trk = len(tracks)

        # construct cost matrix
        cost_mat = np.full((n_trk, n_det), fill_value=1e5, dtype=np.float32)

        for i, trk in enumerate(tracks):
            for j, det in enumerate(det_boxes):
                iou = iou_xyxy(trk.box, det)
                dist = center_dist(trk.box, det)
                cost_mat[i, j] = W_IOU * (1 - iou) +  W_DIST * dist

        # find matches
        row_ind, col_ind = linear_sum_assignment(cost_mat) if cost_mat.size else ([], [])

        
        # 
        assigned_dets = set()
        for r, c in zip(row_ind, col_ind):
            det_box = det_boxes[c]
            if cost_mat[r, c] < (1 - IOU_THR):      # IoU >= IOU_THR
                if tracks[r].check_consistency(det_box):
                    tracks[r].update(det_box)
                    tracks[r].miss_frames = 0
                    assigned_dets.add(c)
                else:
                    tracks[r].update(tracks[r].box_history[-1])  # the most appropriate history point
                    tracks[r].miss_frames = 0
            else:
                print("cost_mat[r, c] < (1 - IOU_THR) is False", cost_mat[r, c], (1 - IOU_THR))
                tracks[r].miss_frames += 1


        # find lost tracks
        still_active = []
        for t in tracks:
            if t.miss_frames >= MAX_MISS:
                lost_tracks.append(t)
            else:
                still_active.append(t)
        tracks = still_active

        # try to reappear lost tracks
        new_lost = []
        for t in lost_tracks:
            recovered = False
            for j, det in enumerate(det_boxes):
                if j in assigned_dets: continue
                if t.check_consistency(det):
                    t.update(det)
                    t.miss_frames = 0
                    t.lost_age = 0
                    tracks.append(t)
                    assigned_dets.add(j)
                    recovered = True
                    break
            if not recovered:
                t.lost_age += 1
                if t.lost_age <= MAX_LOST_AGE:
                    new_lost.append(t)
        lost_tracks = new_lost

        # visualization
        for t in tracks:
            x1,y1,x2,y2 = map(int, t.box)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(frame, f"ID {t.tid}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(os.path.join(OUTPUT_FRAME_DIR, os.path.basename(fpath)), frame)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    print(f"[✓] Tracking finished, video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    video_tractor()


