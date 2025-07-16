from utils import read_video, save_video
from tracker import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import os

# Define paths
VIDEO_PATH = r"C:\Users\LIGHTRQX\Desktop\player_tracking\dataset\15sec_input_720p.mp4"
OUTPUT_DIR = r"C:\Users\LIGHTRQX\Desktop\player_tracking\output\15sec_input_720p.mp4"
MODEL_PATH = r"C:\Users\LIGHTRQX\Desktop\player_tracking\model\best.pt"
STUB_PATH = r"C:\Users\LIGHTRQX\Desktop\player_tracking\stubs\track_stubs.pkl"

def main():
    # Read video
    video_frames = read_video(VIDEO_PATH)

    # Check if video was loaded successfully
    if not video_frames:
        print(f"❌ Failed to load video from: {VIDEO_PATH}")
        return
    else:
        print(f"✅ Loaded {len(video_frames)} frames from: {VIDEO_PATH}")

    # Initialize tracker
    tracker = Tracker(MODEL_PATH)

    # Get object tracks
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=STUB_PATH
    )

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign teams
    team_assigner = TeamAssigner()
    first_frame = video_frames[0]
    first_frame_players = tracks['players'][0]

    team_assigner.assign_team_color(first_frame, first_frame_players)

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]

    # Assign ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')

        if not ball_bbox:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player_track[assigned_player]['has_ball'] = True
            team_ball_control.append(player_track[assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # Draw output frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save output video
    output_path = os.path.join(OUTPUT_DIR, "output_video.avi")
    save_video(output_video_frames, output_path)

    print(f"✅ Output video saved to: {output_path}")

if __name__ == '__main__':
    main()
