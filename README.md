===========================================
⚽ FOOTBALL ANALYSIS PROJECT - CONSOLE VIEW
===========================================

=== OVERVIEW ===
Two AI-driven football analytics tasks:
1. Task 1: Player detection/tracking & cross-camera re-ID
2. Task 2: Match analytics (possession, team clustering, movement metrics)

=== TASK 1: PLAYER TRACKING ===
[GOAL]
- Detect players/referees using YOLOv5
- Track with ByteTrack/BoTSORT
- Cross-camera re-ID with CLIP + Hungarian Matching

[FEATURES]
- Multi-camera support (broadcast + tacticam)
- Real-time annotated output
- Persistent player IDs
- Modular pipeline

[FOLDER STRUCTURE]
player_tracking/
├── task-1.py                 # Main script
├── botsort.yaml              # Config
├── model/best.pt             # YOLO weights
├── dataset/                  # Input videos
│   ├── broadcast.mp4
│   └── tacticam.mp4
└── output/                   # Results
    ├── *_detected/           # Detection frames
    └── *_tracks/             # Tracking data

=== TASK 2: MATCH ANALYTICS ===
[GOAL]
- Team assignment via jersey color clustering
- Ball possession tracking
- Real-world movement estimation
- Speed/distance metrics

[MODULES]
- YOLOv5 (object detection)
- KMeans (jersey clustering)  
- Optical Flow (camera motion)
- Perspective Transform (pixel→meter)
- Speed Calculator

[FOLDER STRUCTURE]
football_analysis/
├── task-2.py                 # Main script
├── utils/                    # Helpers
└── output_videos/            # Results
    ├── possession_map.mp4
    ├── speed_stats.mp4
    └── screenshot.png

=== EXTERNAL FILES NEEDED ===
1. YOLOv5 model: best.pt
   → Download: [drive.google.com/file/d/1DC2k...]
2. Sample match video
   → Download: [drive.google.com/file/d/1t6ag...]

=== FUTURE UPGRADES ===
- 3D pose estimation
- Fatigue detection
- Multi-ball scenarios
- Coaching dashboards

=== USAGE ===
Run either:
$ python player_tracking/task-1.py
or
$ python football_analysis/task-2.py
