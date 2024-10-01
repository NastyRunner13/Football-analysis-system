from utils import read_video, save_video
from trackers import Tracker

def main():
    
    video_frames = read_video('Input_video/08fd33_4.mp4')

    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/tracker.pkl')
    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, 'Output_video/output_video.avi')

if __name__ == "__main__":
    main()