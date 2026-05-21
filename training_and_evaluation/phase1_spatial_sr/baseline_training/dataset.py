import os
import shutil
from pathlib import Path

# Dizinleri Tree'ye göre eşleştirelim
BASE_DIR = Path(".")
SOURCE_TEZ = BASE_DIR / "bitirme tezi"
PUBLISH_2 = BASE_DIR / "publish_2"

# Dosya Eşleştirmeleri (Kaynak -> Hedef)
# Tree çıktına göre doğru yollar:
copy_map = {
    # Faz 2 Kodları
    SOURCE_TEZ
    / "gerçek hayat/faz2/server.py": PUBLISH_2
    / "real_world_inference/phase2_temporal_vfi/server_encoder.py",
    SOURCE_TEZ
    / "gerçek hayat/faz2/client.py": PUBLISH_2
    / "real_world_inference/phase2_temporal_vfi/inference.py",
    # Faz 3 Kodları
    SOURCE_TEZ
    / "gerçek hayat/faz3/client_3.py": PUBLISH_2
    / "real_world_inference/phase3_split_computing/client_decoder.py",
    SOURCE_TEZ
    / "gerçek hayat/faz3/split.py": PUBLISH_2
    / "real_world_inference/phase3_split_computing/split_engine.py",
    SOURCE_TEZ
    / "gerçek hayat/faz3/orjinal.py": PUBLISH_2
    / "real_world_inference/phase3_split_computing/original_analysis.py",
}

print("=" * 60)
print(" 🔄 DOSYALAR YENİDEN VE DOĞRU YERLERE TAŞINIYOR...")
print("=" * 60)

for src, dest in copy_map.items():
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        print(f"[OK] Taşındı: {src.name} -> {dest.relative_to(BASE_DIR)}")
    else:
        print(f"[HATA] Kaynak dosya bulunamadı: {src}")

print("=" * 60)
print(
    " İşlem tamamlandı. 'publish_2' klasöründeki dosya yollarını kontrol edebilirsin."
)
