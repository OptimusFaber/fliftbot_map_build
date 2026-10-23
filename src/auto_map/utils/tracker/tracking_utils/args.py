import argparse


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Demo!")
    parser.add_argument("--video", type=str, help="Путь до файла")
    parser.add_argument("--camera-json", required=True,
                        help="Camera intrinsics sidecar (camera.json)")
    parser.add_argument("--camera-position", required=True,
                        help="Camera mount sidecar: first line cm, second line roll pitch [mount_yaw] radians")
    parser.add_argument("--labels", type=str,
                        help="Optional labels2coords JSON/pickle; omit for tracker-only marker IDs")
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.99, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="fuse_score", default=False, action="store_true",
                        help="fuse score and iou for association")
    parser.add_argument("--cmc-method", default="none", type=str, help="cmc method: files (Vidstab GMC) | orb | ecc")
    # 'proximity_thresh': 0.5
    parser.add_argument("--proximity-thresh", type=float, default=0.5, help="proximity threshold for tracking")
    # 'appearance_thresh': 0.25
    parser.add_argument("--appearance-thresh", type=float, default=0.25, help="appearance threshold for tracking")
    # 'name': None
    parser.add_argument("--name", type=str, default=None, help="name of the tracker (PROBABLY)")  # TODO: Find out what this is used for
    # 'ablation': None
    parser.add_argument("--ablation", type=str, default=None, help="ablation study setting")
    parser.add_argument("--with_reid", default=False, action="store_true", help="whether to use reidentification of bad tracks. Currently not implemented")
    parser.add_argument("--fourier_confidence_thresh", type=float, default=0.7, help="Fourier confidence threshold")
    parser.add_argument("--assignment", type=str, default=None, help="Path to assignment.yaml for adjacency-based code restriction")
    parser.add_argument("--raw-decoder", dest="raw_decoder", default=False, action="store_true",
                        help="Save raw decoder candidates for each detection as codes/confs lists")
    return parser


args = make_parser().parse_args()
