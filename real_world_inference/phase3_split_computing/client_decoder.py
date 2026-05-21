import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import zlib
import time
from collections import deque
import glob
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DECODER_DIR = os.path.join(BASE_DIR, "split_models")
BOTTLENECK_DIR = os.path.join(BASE_DIR, "ablation_analysis")
VIDEO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "dataset", "videos"))

# Testable Configurations
MODELS = {
    "1": {"name": "V5_Standard", "weight": "V5_Standard_decoder.pth"},
    "2": {"name": "V6_Inspector", "weight": "V6_Inspector_decoder.pth"},
}

MODES = {
    "1": {"name": "ZLIB", "ext": "bin"},
    "2": {"name": "CODEC", "ext": "mp4"},
    "3": {"name": "CODEC_ZLIB", "ext": "bin"},
}


# ==========================================
# 2. CLIENT DECODER ARCHITECTURE
# ==========================================
class ClientDecoder(nn.Module):
    def __init__(self):
        super(ClientDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


# ==========================================
# 3. DYNAMIC BOTTLENECK STREAM GENERATOR
# ==========================================
def get_latent_frame_generator(mode, filepath, safe_w, safe_h):
    temp_mp4 = None
    cap = None

    if mode in ["CODEC", "CODEC_ZLIB"]:
        if mode == "CODEC_ZLIB":
            with open(filepath, "rb") as f:
                mp4_bytes = zlib.decompress(f.read())
            temp_mp4 = filepath.replace(".bin", "_TEMP.mp4")
            with open(temp_mp4, "wb") as f:
                f.write(mp4_bytes)
            cap = cv2.VideoCapture(temp_mp4)
        else:
            cap = cv2.VideoCapture(filepath)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()
        if temp_mp4 and os.path.exists(temp_mp4):
            os.remove(temp_mp4)

    elif mode == "ZLIB":
        with open(filepath, "rb") as f:
            data = zlib.decompress(f.read())
        frame_bytes_size = safe_w * safe_h * 3
        num_frames = len(data) // frame_bytes_size

        for i in range(num_frames):
            frame_chunk = data[i * frame_bytes_size : (i + 1) * frame_bytes_size]
            frame_np = np.frombuffer(frame_chunk, dtype=np.uint8).reshape(
                (safe_h, safe_w, 3)
            )
            yield frame_np


# ==========================================
# 4. LIVE PLAYBACK & INFERENCE ENGINE
# ==========================================
def run_interactive_playback(
    video_name, model, model_name, mode_name, mode_ext, device
):
    bottleneck_file = os.path.join(
        BOTTLENECK_DIR, mode_name, f"{video_name}_{model_name}_{mode_name}.{mode_ext}"
    )

    if not os.path.exists(bottleneck_file):
        print(f"\n[-] ERROR: Bottleneck file not found for this configuration!")
        print(f"    Missing Path: {bottleneck_file}")
        return

    # Extract safe width/height from the ground truth video for tensor compatibility
    gt_video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    if not os.path.exists(gt_video_path):
        print(f"\n[-] ERROR: Ground truth video not found: {gt_video_path}")
        return

    gt_cap = cv2.VideoCapture(gt_video_path)
    orig_w = int(gt_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(gt_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    safe_w = orig_w if orig_w % 2 == 0 else orig_w - 1
    safe_h = orig_h if orig_h % 2 == 0 else orig_h - 1
    gt_cap.release()

    frame_gen = get_latent_frame_generator(mode_name, bottleneck_file, safe_w, safe_h)
    time_queue = deque(maxlen=30)
    frame_id = 0

    print(f"\n[🎬] INITIATING LIVE DECODING: {model_name} | MODE: {mode_name}")
    cv2.namedWindow("Phase 3: Interactive Split Computing Engine", cv2.WINDOW_NORMAL)

    for latent_frame in frame_gen:
        input_tensor = (
            torch.from_numpy(latent_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(device)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            reconstructed_tensor = model(input_tensor)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        process_time_ms = (time.time() - start_time) * 1000

        time_queue.append(process_time_ms / 1000)
        avg_time = sum(time_queue) / len(time_queue)
        stable_fps = 1.0 / avg_time if avg_time > 0 else 0

        # Convert tensor to displayable BGR image
        recon_frame = (
            reconstructed_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)
        recon_frame_bgr = cv2.cvtColor(recon_frame, cv2.COLOR_RGB2BGR)

        # Heads-Up Display (HUD) Overlay
        cv2.putText(
            recon_frame_bgr,
            f"SPLIT DECODER: {model_name.upper()}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            recon_frame_bgr,
            f"NETWORK MODE: {mode_name}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3,
        )
        cv2.putText(
            recon_frame_bgr,
            f"LIVE FPS: {int(stable_fps)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            recon_frame_bgr,
            f"FRAME ID: {frame_id}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        cv2.imshow("Phase 3: Interactive Split Computing Engine", recon_frame_bgr)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  [!] Playback interrupted by user.")
            break

    cv2.destroyAllWindows()
    print("[+] Decoding process completed successfully.")


# ==========================================
# 5. INTERACTIVE CLI MENU
# ==========================================
def main():
    os.system("clear" if os.name == "posix" else "cls")
    print("=" * 60)
    print(" 🚀 PHASE 3: INTERACTIVE SPLIT COMPUTING CLIENT")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[!] Hardware Acceleration: {device.type.upper()}\n")

    # 1. Video Selection
    test_videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not test_videos:
        print("[-] ERROR: No 1080p MP4 videos found in 'dataset/videos/' directory.")
        return

    print("[*] Please select a video for testing:")
    for i, vid_path in enumerate(test_videos):
        print(f"  [{i+1}] {os.path.basename(vid_path)}")

    while True:
        try:
            v_choice = int(input("Select Video (e.g., 1): ")) - 1
            if 0 <= v_choice < len(test_videos):
                selected_video_path = test_videos[v_choice]
                video_name = os.path.basename(selected_video_path).replace(".mp4", "")
                break
        except ValueError:
            pass

    # 2. Model Selection
    print("\n[*] Please select the Decoder model:")
    for key, val in MODELS.items():
        print(f"  [{key}] {val['name']}")

    while True:
        m_choice = input("Select Model (e.g., 1): ")
        if m_choice in MODELS:
            selected_model = MODELS[m_choice]
            break

    # 3. Compression Mode Selection
    print("\n[*] Please select the network bottleneck (Compression) mode:")
    for key, val in MODES.items():
        print(f"  [{key}] {val['name']}")

    while True:
        net_choice = input("Select Mode (e.g., 1): ")
        if net_choice in MODES:
            selected_mode = MODES[net_choice]
            break

    # 4. Load Model and Initialize Stream
    weight_path = os.path.join(DECODER_DIR, selected_model["weight"])
    if not os.path.exists(weight_path):
        print(f"\n[-] ERROR: Decoder weight file not found: {weight_path}")
        return

    print(
        f"\n[+] Loading {selected_model['name']} model onto device ({device.type.upper()})..."
    )
    model = ClientDecoder().to(device)
    model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True), strict=False
    )
    model.eval()

    run_interactive_playback(
        video_name,
        model,
        selected_model["name"],
        selected_mode["name"],
        selected_mode["ext"],
        device,
    )


if __name__ == "__main__":
    main()
