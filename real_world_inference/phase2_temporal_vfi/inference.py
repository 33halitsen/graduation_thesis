import cv2
import torch
import torch.nn as nn
import numpy as np
import time, csv, os, json, glob
import threading
import queue
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings

warnings.filterwarnings("ignore")


# ==============================================================================
# 1. MODEL MİMARİSİ (V6: MicroVFINetPro - CoordConv)
# ==============================================================================
class MicroVFINetPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, stride=2), nn.ReLU(True)
        )
        self.drop = nn.Dropout2d(p=0.2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def get_coord_grids(self, x):
        b, c, h, w = x.size()
        y_grid = (
            torch.linspace(-1, 1, h, device=x.device)
            .view(1, 1, h, 1)
            .expand(b, 1, h, w)
        )
        x_grid = (
            torch.linspace(-1, 1, w, device=x.device)
            .view(1, 1, 1, w)
            .expand(b, 1, h, w)
        )
        return torch.cat([x_grid, y_grid], dim=1)

    def forward(self, im1, im3):
        x = torch.cat([im1, im3], dim=1)
        e1 = self.enc1(torch.cat([x, self.get_coord_grids(x)], dim=1))
        e2 = self.drop(self.enc2(e1))
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        return self.final(torch.cat([d1, e1], dim=1))


# ==============================================================================
# 2. KUYRUK VE OYNATMA SİSTEMİ (JITTER BUFFER)
# ==============================================================================
class VideoPlayer:
    def __init__(self, target_fps=25, log_csv_path="playback_fps_log.csv"):
        self.frame_queue = queue.Queue(maxsize=300)
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        self.inference_done = False  # Worker'ın işini bitirip bitirmediğini takip eder
        self.force_quit = False  # Kullanıcı Q'ya basarsa sistemi komple durdurur
        self.current_savings = 0.0
        self.log_csv_path = log_csv_path

    def start_playback(self):
        window_name = "Faz 2: Constant FPS VFI Player"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1920, 1080)
        print(f"🎬 Oynatıcı Başlatıldı ({self.target_fps} FPS Sabitleyici Aktif)")

        log_dir = os.path.dirname(self.log_csv_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(self.log_csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Timestamp", "Gercek_FPS", "Kuyruk_Dolulugu", "Durum"])

            last_frame_time = time.time()

            # Oynatıcı, kullanıcı çıkış yapmadığı sürece VE
            # (Inference devam ediyorsa VEYA kuyrukta hala kare varsa) çalışmaya devam eder.
            while not self.force_quit and (
                not self.inference_done or not self.frame_queue.empty()
            ):
                try:
                    # FPS kaybını önlemek için Timeout süresi
                    frame_data = self.frame_queue.get(timeout=0.05)
                    frame, tag, color, coords, sc = frame_data

                    current_time = time.time()
                    elapsed_since_last = current_time - last_frame_time

                    actual_fps = (
                        1.0 / elapsed_since_last
                        if elapsed_since_last > 0
                        else self.target_fps
                    )
                    last_frame_time = current_time

                    h, w = frame.shape[:2]
                    disp = cv2.resize(frame, (int(w * sc), int(h * sc)))

                    q_size = self.frame_queue.qsize()

                    # Bilgilendirme Metinleri
                    cv2.putText(disp, f"VFI V6 | {tag}", (20, 40), 1, 2, color, 2)
                    cv2.putText(
                        disp,
                        f"Hedef FPS: {self.target_fps}",
                        (20, 80),
                        1,
                        1.5,
                        (255, 255, 255),
                        2,
                    )

                    fps_color = (
                        (0, 255, 0)
                        if actual_fps >= self.target_fps * 0.9
                        else (0, 165, 255)
                    )
                    cv2.putText(
                        disp,
                        f"Gercek FPS: {actual_fps:.1f}",
                        (20, 120),
                        1,
                        1.5,
                        fps_color,
                        2,
                    )
                    cv2.putText(
                        disp,
                        f"Tampon (Kuyruk): {q_size} kare",
                        (20, 160),
                        1,
                        1.5,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        disp,
                        f"Tasarruf: %{self.current_savings}",
                        (20, 200),
                        1,
                        1.5,
                        (0, 255, 0),
                        2,
                    )

                    if coords and tag == "AI_SENTEZ":
                        cx1, cy1, cx2, cy2 = [int(c * sc) for c in coords]
                        cv2.rectangle(disp, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)

                    cv2.imshow("Faz 2: Constant FPS VFI Player", disp)

                    # Loglama
                    writer.writerow(
                        [
                            time.strftime("%H:%M:%S.%f")[:-3],
                            f"{actual_fps:.2f}",
                            q_size,
                            "OYNATILIYOR",
                        ]
                    )

                    process_time = time.time() - current_time
                    wait_time = max(1, int((self.frame_duration - process_time) * 1000))

                    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                        self.force_quit = True
                        break

                except queue.Empty:
                    # Eğer inference bittiyse ve kuyruk boşsa çık
                    if self.inference_done:
                        break

                    print(
                        "⚠️ UYARI: Kuyruk boşaldı! AI işlemleri ekrana yetişemiyor (Buffer Underrun)."
                    )
                    writer.writerow(
                        [
                            time.strftime("%H:%M:%S.%f")[:-3],
                            "0.00",
                            0,
                            "DARBOGAZ (DONMA)",
                        ]
                    )
                    continue

        cv2.destroyAllWindows()


# ==============================================================================
# 3. İŞLEMCİ MOTORU (ARKA PLAN - THREAD)
# ==============================================================================
def inference_worker(video_path, manifest_path, model, player, device, out_csv):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    player.current_savings = manifest["savings_percent"]

    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    cap.release()

    # PSNR için kareleri RAM'de tutacağımız geçici liste
    frames_for_evaluation = []

    for op in manifest["operations"]:
        if player.force_quit:
            break

        s, e, coords = op["start"], op["end"], op["coords"]
        batch = {
            s: all_frames[s].copy(),
            e: (all_frames[e].copy() if e < len(all_frames) else all_frames[-1].copy()),
        }

        # AI İşlemi
        if op["action"].startswith("SKIP"):
            hierarchical_inference(model, batch, s, e, coords, device)

        for k in range(s, e):
            out_f = batch.get(k, all_frames[k].copy())
            is_ai = k != s and not op["action"] == "SEND_ALL"

            tag = "AI_SENTEZ" if is_ai else "SUNUCU_KARE"
            color = (0, 255, 0) if is_ai else (0, 0, 255)

            h, w = out_f.shape[:2]
            sc = 720 / h if h > w else 1280 / w

            # Sadece kuyruğa yolla, FPS testi sekteye uğramasın
            player.frame_queue.put((out_f.copy(), tag, color, coords, sc))

            # Aşama 2 (PSNR Testi) için listeye ekle
            frames_for_evaluation.append((k, tag, is_ai, out_f.copy()))

    # İşçi bitti sinyali ver (Kuyruk boşalana kadar player devam edecek)
    player.inference_done = True
    print("✅ Aşama 1: Yapay Zeka çıkarımı ve Kuyruk doldurma tamamlandı.")

    # ==========================================
    # AŞAMA 2: PSNR HESAPLAMASI (Oynatma'dan bağımsız)
    # ==========================================
    if not player.force_quit:
        print("⏳ Aşama 2: PSNR Akademik Metrikleri hesaplanıyor, lütfen bekleyin...")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Kare", "Tip", "PSNR", "Bandwidth"])

            for k, tag, is_ai, out_f in frames_for_evaluation:
                # Ağır işlem artık burada yapılıyor
                p_val = psnr(all_frames[k], out_f, data_range=255)
                writer.writerow(
                    [k, tag, f"{p_val:.2f}", "SAVED" if is_ai else "DOWNLOADED"]
                )
        print(
            f"✅ Aşama 2: PSNR metrikleri başarıyla '{os.path.basename(out_csv)}' dosyasına kaydedildi."
        )


# --- Yardımcı Fonksiyonlar ---
def apply_blending(bg_img, patch_img, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    mask = np.zeros((h, w), dtype=np.float32)
    mask[
        max(1, int(h * 0.1)) : -max(1, int(h * 0.1)),
        max(1, int(w * 0.1)) : -max(1, int(w * 0.1)),
    ] = 1.0
    mask = np.stack([cv2.GaussianBlur(mask, (21, 21), 11)] * 3, axis=-1)
    res = bg_img.copy()
    res[y1:y2, x1:x2] = (
        (patch_img.astype(np.float32) * mask)
        + (bg_img[y1:y2, x1:x2].astype(np.float32) * (1.0 - mask))
    ).astype(np.uint8)
    return res


def predict_middle_frame(model, img1, img3, coords, device):
    if coords is None or (coords[0] >= coords[2] or coords[1] >= coords[3]):
        return img1.copy()
    x1, y1, x2, y2 = coords
    p1 = cv2.cvtColor(img1[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    p3 = cv2.cvtColor(img3[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    t1 = torch.from_numpy(p1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    t3 = torch.from_numpy(p3).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        pred = model(t1, t3)
    res_p = (
        (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return apply_blending(img1, cv2.cvtColor(res_p, cv2.COLOR_RGB2BGR), x1, y1, x2, y2)


def hierarchical_inference(model, frames, s, e, coords, device):
    if e - s <= 1:
        return
    mid = s + (e - s) // 2
    frames[mid] = predict_middle_frame(model, frames[s], frames[e], coords, device)
    hierarchical_inference(model, frames, s, mid, coords, device)
    hierarchical_inference(model, frames, mid, e, coords, device)


# ==============================================================================
# 4. ANA ÇALIŞTIRICI
# ==============================================================================
if __name__ == "__main__":
    BASE = "/Users/halitsen/Desktop/graduation_thesis"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = f"{BASE}/real_world_inference/phase2_temporal_vfi/weights/best_micro_vfi_pro_v6.pth"

    model = MicroVFINetPro().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    VIDEO_MAP = {
        "video1": "12331098_1080_1920_25fps.mp4",
        "video2": "15439560_1920_1080_25fps.mp4",
    }

    for v_key in ["video1", "video2"]:
        v_name = v_key.capitalize() 
        
        m_p = f"{BASE}/real_world_inference/phase2_temporal_vfi/manifests/Manifest_{v_name}.json"
        v_p = f"{BASE}/dataset/videos/{VIDEO_MAP[v_key]}"
        
        out_csv = (
            f"{BASE}/real_world_inference/phase2_temporal_vfi/numerical_reports/Akademik_V6_Final_{v_name}.csv"
        )
        playback_csv = (
            f"{BASE}/real_world_inference/phase2_temporal_vfi/numerical_reports/Playback_FPS_{v_name}.csv"
        )

        if not os.path.exists(m_p):
            print(f"⚠️ Manifest not found: {m_p}")
        if not os.path.exists(v_p):
            print(f"⚠️ Video not found: {v_p}")

        if os.path.exists(m_p) and os.path.exists(v_p):
            player = VideoPlayer(target_fps=25, log_csv_path=playback_csv)

            worker_thread = threading.Thread(
                target=inference_worker, args=(v_p, m_p, model, player, device, out_csv)
            )
            worker_thread.start()

            player.start_playback()

            worker_thread.join()
            print(f"--- {v_name} Testi Bitti ---")