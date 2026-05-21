import cv2
import numpy as np
import os
import glob
import json
import time

# ==============================================================================
# 🚀 MICRO-VFI PRO V6: SUNUCU (SERVER) OPTİK AKIŞ VE MANİFESTO MOTORU
# ==============================================================================

# 1. SINIRLAR VE EŞİKLER (M1 Hız ve Literatür Testlerinden Alındı)
MAX_PATCH_AREA = 768 * 768  # M1 için 20 FPS altına düşmeme sınırı (23.5 FPS)
MAX_DISPLACEMENT = 8.0  # Halüsinasyon ve ghosting (makro hareket) sınırı
MIN_CONTOUR_AREA = 25  # Kamera karıncalanması (noise) filtresi
STRIDE_CHECK = 4  # Tenzör boyut uyumluluğu için katsayı


def analyze_movement(img1, img2):
    """İki kare arasındaki hareketi analiz eder, sınırları test eder ve yama koordinatlarını döndürür."""
    h_orig, w_orig = img1.shape
    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA
    ]

    # Eğer hiç hareket yoksa (Durağan sahne), devasa tasarruf! 0x0 yama ile onay ver.
    if not valid_contours:
        return True, [0, 0, 0, 0], 0.0

    # Tüm hareketli alanları kapsayan en küçük dikdörtgeni bul (Bounding Box)
    x, y, w, h = cv2.boundingRect(np.concatenate(valid_contours))

    # Modelin Stride uyumluluğu için yama boyutunu 4'ün katlarına yuvarla
    nw = ((w + STRIDE_CHECK - 1) // STRIDE_CHECK) * STRIDE_CHECK
    nh = ((h + STRIDE_CHECK - 1) // STRIDE_CHECK) * STRIDE_CHECK
    nx = max(0, x - (nw - w) // 2)
    ny = max(0, y - (nh - h) // 2)

    if nx + nw > w_orig:
        nx = w_orig - nw
    if ny + nh > h_orig:
        ny = h_orig - nh

    patch_area = nw * nh

    # KURAL 1: Donanım Hız Sınırı (Apple M1 768x768'den büyüğünde 13 FPS'e çakılır)
    if patch_area > MAX_PATCH_AREA:
        return False, None, None

    # KURAL 2: Optik Akış (Kalite) Sınırı
    patch_thresh = thresh[ny : ny + nh, nx : nx + nw]
    patch1 = img1[ny : ny + nh, nx : nx + nw]
    patch2 = img2[ny : ny + nh, nx : nx + nw]

    # Farneback ile piksel kayma miktarını (Magnitude) hesapla
    flow = cv2.calcOpticalFlowFarneback(patch1, patch2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    valid_mag = mag[patch_thresh > 0]
    if len(valid_mag) == 0:
        return True, [int(nx), int(ny), int(nx + nw), int(ny + nh)], 0.0

    avg_displacement = np.mean(valid_mag)

    # Eğer hareket 8.0 pikselden büyükse model halüsinasyon görür, iptal et!
    if avg_displacement > MAX_DISPLACEMENT:
        return False, None, None

    return True, [int(nx), int(ny), int(nx + nw), int(ny + nh)], avg_displacement


def process_video_server(video_path, video_name, out_json):
    print(f"\n[📡 SUNUCU] Video Analiz Ediliyor: {video_name}")
    cap = cv2.VideoCapture(video_path)
    frames_gray = []

    # Sunucu tarafında tüm kareleri gri tonda belleğe alıyoruz (Hızlı random-access için)
    print(" -> Kareler belleğe alınıyor (Optik Akış için)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    total_frames = len(frames_gray)
    if total_frames < 10:
        print(" -> Video çok kısa, atlanıyor.")
        return

    i = 0
    operations = []
    sent_frames_count = 1  # İlk kare her zaman gönderilir

    print(f" -> Hiyerarşik Karar Ağacı Çalışıyor (Toplam {total_frames} Kare)...")

    # Zaman ölçümü
    start_time = time.time()

    while i < total_frames - 1:
        # 1. DENEME: 7 Kare Atlama (F1 -> F9)
        if i + 8 < total_frames:
            is_valid, coords, disp = analyze_movement(
                frames_gray[i], frames_gray[i + 8]
            )
            if is_valid:
                operations.append(
                    {
                        "start": i,
                        "end": i + 8,
                        "action": "SKIP_7",
                        "coords": coords,
                        "disp_px": round(float(disp), 2),
                    }
                )
                sent_frames_count += 1  # Sadece i+8 karesi gönderilecek
                i += 8
                continue

        # 2. DENEME: 3 Kare Atlama (F1 -> F5)
        if i + 4 < total_frames:
            is_valid, coords, disp = analyze_movement(
                frames_gray[i], frames_gray[i + 4]
            )
            if is_valid:
                operations.append(
                    {
                        "start": i,
                        "end": i + 4,
                        "action": "SKIP_3",
                        "coords": coords,
                        "disp_px": round(float(disp), 2),
                    }
                )
                sent_frames_count += 1
                i += 4
                continue

        # 3. DENEME: 1 Kare Atlama (F1 -> F3)
        if i + 2 < total_frames:
            is_valid, coords, disp = analyze_movement(
                frames_gray[i], frames_gray[i + 2]
            )
            if is_valid:
                operations.append(
                    {
                        "start": i,
                        "end": i + 2,
                        "action": "SKIP_1",
                        "coords": coords,
                        "disp_px": round(float(disp), 2),
                    }
                )
                sent_frames_count += 1
                i += 2
                continue

        # 4. SON ÇARE: İptal Et, Orijinali Gönder (F1 -> F2)
        # Aşırı hareket var veya yama çok büyük. Yapay Zeka çöker, doğrudan orijinal kareyi gönderiyoruz.
        operations.append(
            {
                "start": i,
                "end": i + 1,
                "action": "SEND_ALL",
                "coords": None,
                "disp_px": None,
            }
        )
        sent_frames_count += 1
        i += 1

    calc_time = time.time() - start_time

    # BANT GENİŞLİĞİ TASARRUFU HESAPLAMASI
    saved_frames = total_frames - sent_frames_count
    savings_percent = (saved_frames / total_frames) * 100.0

    manifest_data = {
        "video_name": video_name,
        "total_frames_original": total_frames,
        "frames_to_transmit": sent_frames_count,
        "frames_saved_by_ai": saved_frames,
        "bandwidth_savings_percent": round(savings_percent, 2),
        "server_processing_time_sec": round(calc_time, 2),
        "operations": operations,
    }

    with open(out_json, "w") as f:
        json.dump(manifest_data, f, indent=4)

    print(
        f" [✅] Manifest Oluşturuldu! Tasarruf: %{savings_percent:.2f} ({saved_frames} kare ağa çıkmaktan kurtarıldı)"
    )


if __name__ == "__main__":
    BASE_DIR = "/Users/halitsen/Desktop/tez/bitirme tezi"

    # Faz 2 klasör yapılandırması
    faz2_dir = os.path.join(BASE_DIR, "gerçek hayat", "faz2")
    manifest_dir = os.path.join(faz2_dir, "manifestler")
    os.makedirs(manifest_dir, exist_ok=True)

    videolar = ["video1", "video2", "video3"]
    res = "1080p"  # VFI işlemlerini orijinal yüksek çözünürlük üzerinden test ediyoruz (veya isteğe göre 360p yapılabilir)

    print("=" * 60)
    print("   🌐 FAZ 2: SUNUCU MANİFESTO ÜRETİCİSİ BAŞLATILDI")
    print("=" * 60)

    for video in videolar:
        lr_pattern = (
            f"{BASE_DIR}/gerçek hayat/videolar/dusuk_cozunurluklu/{video}/*_{res}.mp4"
        )
        try:
            VIDEO_PATH = glob.glob(lr_pattern)[0]
        except IndexError:
            # Düşük çözünürlüklü klasöründe 1080p yoksa, ana dizindeki orijinali dene
            hr_isim = f"{video}.mp4"  # İsmi dinamik olarak ana klasörden çekmek gerekebilir, basit tutuyoruz.
            # Tree'ye göre: "12331098_1080_1920_25fps.mp4" vb.
            pattern_hr = (
                f"{BASE_DIR}/gerçek hayat/videolar/*_{video.replace('video','')}*.mp4"
            )
            bulunan = glob.glob(
                f"{BASE_DIR}/gerçek hayat/videolar/dusuk_cozunurluklu/{video}/*_{res}.mp4"
            )
            if not bulunan:
                print(f"  [-] {video} için {res} video bulunamadı, atlanıyor.")
                continue
            VIDEO_PATH = bulunan[0]

        out_json = os.path.join(manifest_dir, f"manifest_{video}.json")
        process_video_server(VIDEO_PATH, video, out_json)

    print("\n" + "=" * 60)
    print(f" 🎉 TÜM MANİFESTOLAR HAZIR! Rota: {manifest_dir}")
    print(
        " Artık istemci (Client) kodu bu JSON'ları okuyup doğrudan modeli çalıştırabilir."
    )
    print("=" * 60)
