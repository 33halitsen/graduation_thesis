import cv2
import torch
import numpy as np
import time
import csv
import os
import glob
from PIL import Image
from models import ESPCN_Baseline, FastResESPCN, ESPCN_Hybrid
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "videos", "low_resolution")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "numerical_reports")
SCALE_FACTOR = 3

MODELS_TO_TEST = {
    "baseline": {"class": ESPCN_Baseline, "weight": "espcn_baseline.pth"},
    "fastres": {"class": FastResESPCN, "weight": "fast_res_espcn_best.pth"},
    "hybrid_v5": {"class": ESPCN_Hybrid, "weight": "espcn_hybrid_v5.pth"},
}


def run_automated_fps_test(lr_video, model, model_name, device):
    video_name = os.path.basename(lr_video)
    output_csv = os.path.join(OUTPUT_DIR, f"FPS_Report_{model_name}_{video_name}.csv")

    cap = cv2.VideoCapture(lr_video)
    fps_list = []
    target_w, target_h = None, None
    frame_idx = 0

    print(f"\n[*] Testing {model_name.upper()} on {video_name}...")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame_No", "Process_Time_ms", "Instant_FPS"])

        while True:
            ret, lr_frame = cap.read()
            if not ret:
                break

            if target_w is None:
                target_w = lr_frame.shape[1] * SCALE_FACTOR
                target_h = lr_frame.shape[0] * SCALE_FACTOR

            start_time = time.time()
            lr_rgb = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)

            if model_name == "baseline":
                rgb_tensor = (
                    torch.from_numpy(lr_rgb)
                    .float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )
                with torch.no_grad():
                    sr_tensor = model(rgb_tensor)
                sr_rgb = np.clip(
                    sr_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                sr_frame = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
            else:
                pil_img = Image.fromarray(lr_rgb)
                y_pil, cb_pil, cr_pil = pil_img.convert("YCbCr").split()
                y_tensor = (
                    torch.from_numpy(np.array(y_pil))
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )
                with torch.no_grad():
                    sr_y_tensor = model(y_tensor)
                sr_y_np = np.clip(
                    sr_y_tensor.squeeze().cpu().numpy() * 255.0, 0, 255
                ).astype(np.uint8)
                cb_resized = cb_pil.resize((target_w, target_h), resample=Image.BICUBIC)
                cr_resized = cr_pil.resize((target_w, target_h), resample=Image.BICUBIC)
                hybrid_ycbcr = Image.merge(
                    "YCbCr", (Image.fromarray(sr_y_np), cb_resized, cr_resized)
                )
                sr_frame = cv2.cvtColor(
                    np.array(hybrid_ycbcr.convert("RGB")), cv2.COLOR_RGB2BGR
                )

            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            fps_list.append(fps)
            writer.writerow([frame_idx, f"{process_time*1000:.2f}", f"{fps:.2f}"])

            cv2.putText(
                sr_frame,
                f"[{model_name.upper()}] FPS: {fps:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow(
                "Interactive Inference Engine",
                cv2.resize(sr_frame, (sr_frame.shape[1] // 2, sr_frame.shape[0] // 2)),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  [!] Test kullanıcı tarafından durduruldu.")
                break

            if frame_idx % 50 == 0:
                print(f"  -> Processed Frame: {frame_idx} | Current FPS: {fps:.1f}")
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    with open(output_csv, mode="a", newline="") as file:
        writer.writerow(["---", "---", "---"])
        writer.writerow(["AVERAGE_FPS", "", f"{avg_fps:.2f}"])
    print(f"[+] Average FPS: {avg_fps:.2f} -> Saved to {os.path.basename(output_csv)}")


def main():
    print("=" * 50)
    print(" 🚀 PHASE 1: INTERACTIVE SPATIAL SUPER RESOLUTION")
    print("=" * 50)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[!] Hardware Acceleration: {device.type.upper()}\n")

    test_videos = glob.glob(
        os.path.join(DATASET_DIR, "**", "*_360p.mp4"), recursive=True
    )
    if not test_videos:
        print("[-] ERROR: No 360p videos found in dataset folder.")
        return

    print("[*] Lütfen test etmek istediğiniz videoyu seçin:")
    for i, vid_path in enumerate(test_videos):
        print(f"  [{i+1}] {os.path.basename(vid_path)}")

    while True:
        try:
            vid_choice = int(input("Video Seçiminiz (Örn: 1): ")) - 1
            if 0 <= vid_choice < len(test_videos):
                selected_video = test_videos[vid_choice]
                break
            else:
                print(
                    "[-] Geçersiz numara. Lütfen listedeki numaralardan birini girin."
                )
        except ValueError:
            print("[-] Lütfen sadece rakam girin.")

    model_keys = list(MODELS_TO_TEST.keys())
    print("\n[*] Lütfen kullanmak istediğiniz modeli seçin:")
    for i, m_key in enumerate(model_keys):
        print(f"  [{i+1}] {m_key.upper()} (Ağırlık: {MODELS_TO_TEST[m_key]['weight']})")

    while True:
        try:
            model_choice = int(input("Model Seçiminiz (Örn: 1): ")) - 1
            if 0 <= model_choice < len(model_keys):
                selected_model_name = model_keys[model_choice]
                selected_model_info = MODELS_TO_TEST[selected_model_name]
                break
            else:
                print(
                    "[-] Geçersiz numara. Lütfen listedeki numaralardan birini girin."
                )
        except ValueError:
            print("[-] Lütfen sadece rakam girin.")

    weight_path = os.path.join(WEIGHTS_DIR, selected_model_info["weight"])
    if not os.path.exists(weight_path):
        print(
            f"\n[-] ERROR: Ağırlık dosyası bulunamadı ({weight_path}). Lütfen ağırlıkların klasörde olduğundan emin olun."
        )
        return

    print(f"\n[+] Model yükleniyor: {selected_model_name.upper()}...")
    model = selected_model_info["class"](upscale_factor=SCALE_FACTOR).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    run_automated_fps_test(selected_video, model, selected_model_name, device)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
