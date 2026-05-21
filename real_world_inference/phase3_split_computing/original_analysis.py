import os
import csv

# --- 1. DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Orijinal Videoların Yeri
VIDEO_DIR = os.path.abspath(os.path.join(BASE_DIR, "../videolar"))

# Kaggle'dan inen darboğaz dosyalarının yeri
BOTTLENECK_DIR = os.path.join(BASE_DIR, "kaggle/ablation_sistemi")

# Çıktı Raporu Yeri
REPORTS_DIR = os.path.join(BASE_DIR, "test_sonuclari")
os.makedirs(REPORTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(REPORTS_DIR, "Net_Boyut_Kuculme_Analizi.csv")

# Test Edilen Veriler
VIDEOS = [
    "12331098_1080_1920_25fps",
    "15439560_1920_1080_25fps",
    "15773739_1920_1080_25fps",
]
MODELS = ["V5_Standart", "V6_Mufettisli"]
MODES = ["ZLIB", "CODEC", "CODEC_ZLIB"]

# --- 2. ANALİZ MOTORU ---
print("=" * 60)
print(" SADECE BOYUT VE SIKIŞTIRMA (KÜÇÜLME) ANALİZİ")
print("=" * 60)

with open(CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Rapor Başlıkları
    writer.writerow(
        [
            "Video_Adi",
            "Model",
            "Aktarim_Modu",
            "Orijinal_Video_MB",
            "DarBogaz_Dosyasi_MB",
            "Tasarruf_Orani_%",
        ]
    )

    for video in VIDEOS:
        orig_video_path = os.path.join(VIDEO_DIR, f"{video}.mp4")

        # Orijinal video kontrolü
        if not os.path.exists(orig_video_path):
            print(f"[HATA] Orijinal video bulunamadı: {orig_video_path}")
            continue

        # Orijinal boyutu MB cinsinden oku
        orig_size_mb = os.path.getsize(orig_video_path) / (1024 * 1024)
        print(f"\n[ORİJİNAL HEDEF] {video} -> Boyut: {orig_size_mb:.2f} MB")

        for model in MODELS:
            for mode in MODES:
                # CODEC modu mp4, diğerleri bin uzantılı
                ext = "mp4" if mode == "CODEC" else "bin"
                bottleneck_path = os.path.join(
                    BOTTLENECK_DIR, mode, f"{video}_{model}_{mode}.{ext}"
                )

                if os.path.exists(bottleneck_path):
                    # Darboğaz boyutunu MB cinsinden oku
                    bottleneck_size_mb = os.path.getsize(bottleneck_path) / (
                        1024 * 1024
                    )

                    # Tasarruf Oranı (Küçülme Yüzdesi) Hesabı
                    tasarruf_orani = (1 - (bottleneck_size_mb / orig_size_mb)) * 100

                    # CSV'ye yaz
                    writer.writerow(
                        [
                            video,
                            model,
                            mode,
                            round(orig_size_mb, 2),
                            round(bottleneck_size_mb, 2),
                            round(tasarruf_orani, 2),
                        ]
                    )

                    print(
                        f"  -> {model} | {mode} | Darboğaz: {bottleneck_size_mb:.2f} MB | Tasarruf: %{tasarruf_orani:.2f}"
                    )
                else:
                    print(f"  [EKSİK DOSYA] -> {os.path.basename(bottleneck_path)}")

print("=" * 60)
print(f"[TAMAMLANDI] Net Boyut Analiz Raporu oluşturuldu:\n -> {CSV_PATH}")
print("=" * 60)
