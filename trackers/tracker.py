from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        
        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}


            detection_supervision = sv.Detections.from_ultralytics(detection)
   
            for object_idx in range(len(detection_supervision)):
                class_id = detection_supervision.class_id[object_idx]
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for bbox, confidence, class_id, tracker_id in zip(
                detection_with_tracks.xyxy,
                detection_with_tracks.confidence,
                detection_with_tracks.class_id,
                detection_with_tracks.tracker_id,
            ):
                bbox = bbox.tolist()
                cls_id = class_id
                track_id = tracker_id

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Loop over detections without tracks (for the ball)
            for bbox, class_id in zip(
                detection_supervision.xyxy,
                detection_supervision.class_id,
            ):
                bbox = bbox.tolist()
                if class_id == cls_names_inv["ball"]:
                    # Assuming ball has a unique track ID
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frames, bbox, color, track_id):
        y2 = int(bbox[3])
    
    def draw_annotations(self, video_frames, tracks):

        output_video_frame = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id)
