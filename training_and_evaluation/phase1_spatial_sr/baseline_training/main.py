import os
from config import Config
from evaluator import Evaluator


def run_tests():
    evaluator = Evaluator()

    if not os.path.exists(Config.TEST_IMAGE_DIR):
        os.makedirs(Config.TEST_IMAGE_DIR)
        print(
            f"[INFO] Created {Config.TEST_IMAGE_DIR}. Please put some 1080p images there."
        )
        return

    test_images = [
        f
        for f in os.listdir(Config.TEST_IMAGE_DIR)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    if not test_images:
        print("[WARNING] No test images found in the data/test_images folder.")
        return

    print(f"[INFO] Found {len(test_images)} images for testing. Starting evaluation...")

    total_psnr = 0
    total_ssim = 0

    for idx, img_name in enumerate(test_images):
        img_path = os.path.join(Config.TEST_IMAGE_DIR, img_name)
        output_name = f"result_{idx+1:03d}_{img_name}"

        psnr_val, ssim_val = evaluator.evaluate_image(img_path, output_name)
        total_psnr += psnr_val
        total_ssim += ssim_val

    avg_psnr = total_psnr / len(test_images)
    avg_ssim = total_ssim / len(test_images)
    print("\n" + "=" * 40)
    print(f"TEST COMPLETED. AVERAGE ESPCN METRICS:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    run_tests()
