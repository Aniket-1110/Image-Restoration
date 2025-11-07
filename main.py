import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

try:
    from skimage import restoration
except Exception as e:
    restoration = None

NOISE_STD_THRESHOLD = 10.0       
BLUR_VAR_THRESHOLD = 100.0       
CONTRAST_STD_THRESHOLD = 40.0    
DAMAGED_PIXELS_FRACTION = 0.02
DAMAGE_INTENSITY_THRESHOLD = 10
RL_SHARPNESS_IMPROVEMENT_FACTOR = 1.05  

def ensure_dir_for_file(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def show_bgr(img, title=""):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6)); plt.imshow(rgb); plt.axis("off"); plt.title(title); plt.show()

#detect
def estimate_noise_std(gray):
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return residual.std()

def estimate_blur_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def estimate_contrast_std(gray):
    return gray.std()

def detect_damage(gray):
    h, w = gray.shape
    total = h * w
    low_mask = (gray <= DAMAGE_INTENSITY_THRESHOLD)
    high_mask = (gray >= 255 - DAMAGE_INTENSITY_THRESHOLD)
    combined = (low_mask | high_mask).astype(np.uint8) * 255
    frac = (combined.sum() / 255) / total
    return (frac >= DAMAGED_PIXELS_FRACTION), combined

#restore
def denoise_nlmeans(img_bgr, h=10.0, hColor=10.0):
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, hColor, 7, 21)

def clahe_color(img_bgr, clipLimit=2.0, tileGridSize=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def sharpen_unsharp(img_bgr, kernel_size=(5,5), sigma=1.0, amount=1.2, threshold=10):
    blurred = cv2.GaussianBlur(img_bgr, kernel_size, sigma)
    sharpened = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        diff = np.abs(img_bgr.astype(np.int16) - blurred.astype(np.int16))
        mask = np.any(diff > threshold, axis=2)
        out = img_bgr.copy()
        out[mask] = sharpened[mask]
        return out
    return sharpened

def inpaint_img(img_bgr, mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

def gaussian_psf(kernel_size, sigma):
    k = kernel_size
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def richardson_lucy_deblur(img_bgr, psf, iterations=60, clip=True):
    from skimage import restoration
    import numpy as np
    img_float = img_bgr.astype(np.float32) / 255.0
    deconv = np.zeros_like(img_float)
    for c in range(3): 
        channel = np.clip(img_float[..., c], 1e-8, 1.0)
        deconv[..., c] = restoration.richardson_lucy(channel, psf, num_iter=iterations, clip=clip)
    deconv_img = np.clip(deconv * 255.0, 0, 255).astype(np.uint8)
    return deconv_img

def sharpness_metric(gray):
    return estimate_blur_variance(gray)  


#main
def auto_restore(img_bgr, debug=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    noise_std = estimate_noise_std(gray)
    blur_var = estimate_blur_variance(gray)
    contrast_std = estimate_contrast_std(gray)
    damaged, damage_mask = detect_damage(gray)

    decisions = {
        "noisy": noise_std > NOISE_STD_THRESHOLD,
        "blurry": blur_var < BLUR_VAR_THRESHOLD,
        "low_contrast": contrast_std < CONTRAST_STD_THRESHOLD,
        "damaged": damaged
    }

    if debug:
        print("\n[Image Analysis]")
        print(f" Noise STD: {noise_std:.2f}  (>{NOISE_STD_THRESHOLD} -> noisy={decisions['noisy']})")
        print(f" Blur Var : {blur_var:.2f}  (<{BLUR_VAR_THRESHOLD} -> blurry={decisions['blurry']})")
        print(f" Contrast : {contrast_std:.2f}  (<{CONTRAST_STD_THRESHOLD} -> low_contrast={decisions['low_contrast']})")
        print(f" Damage Detected: {decisions['damaged']}")

    result = img_bgr.copy()
    steps = []

    if decisions["damaged"]:
        result = inpaint_img(result, damage_mask)
        steps.append("inpaint")
        if debug: print("Applied: inpaint")

    if decisions["noisy"]:
        h = max(8.0, noise_std * 0.6) 
        result = denoise_nlmeans(result, h=h, hColor=h)
        steps.append("denoise")
        if debug: print(f"Applied: denoise (h={h:.1f})")

    gray_before_deblur = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    sharp_before = sharpness_metric(gray_before_deblur)

    if decisions["blurry"]:
        if restoration is None:
            if debug:
                print("skimage not available — skipping RL deconvolution and using unsharp mask.")
            result = sharpen_unsharp(result, amount=1.4, sigma=1.2, threshold=12)
            steps.append("sharpen_unsharp")
        else:
            est_sigma = np.clip(max(1.0, 40.0 / (blur_var + 1e-6)), 1.0, 4.5)
            ksize = int(min(41, max(7, int(est_sigma * 8))))
            if ksize % 2 == 0:
                ksize += 1
            psf = gaussian_psf(ksize, est_sigma)

            RL_ITERS = 50
            if debug:
                print(f"RL deconvolution attempt: est_sigma={est_sigma:.2f}, psf_size={ksize}, iterations={RL_ITERS}")

            try:
                rl_candidate = richardson_lucy_deblur(result, psf, iterations=RL_ITERS, clip=True)
                rl_gray = cv2.cvtColor(rl_candidate, cv2.COLOR_BGR2GRAY)
                sharp_after_rl = sharpness_metric(rl_gray)

                if debug:
                    print(f" Sharpness before RL: {sharp_before:.2f}, after RL: {sharp_after_rl:.2f}")
                    
                if sharp_after_rl >= sharp_before * RL_SHARPNESS_IMPROVEMENT_FACTOR:
                    result = rl_candidate
                    steps.append("richardson_lucy")
                    if debug:
                        print("Applied: Richardson-Lucy deconvolution (accepted)")
                else:
                    result = sharpen_unsharp(result, amount=1.6, sigma=1.5, threshold=12)
                    steps.append("sharpen_unsharp")
                    if debug:
                        print("RL did not improve sharpness enough. Fallback: applied unsharp mask")
            except Exception as e:
                if debug:
                    print("RL deconvolution failed:", str(e))
                result = sharpen_unsharp(result, amount=1.4, sigma=1.2, threshold=12)
                steps.append("sharpen_unsharp")

    else:
        if "denoise" in steps:
            result = sharpen_unsharp(result, amount=1.0, sigma=1.0, threshold=12)
            steps.append("sharpen_unsharp")
            if debug:
                print("Not flagged blurry but denoised earlier — applied mild unsharp")

    if decisions["low_contrast"]:
        result = clahe_color(result)
        steps.append("equalize")
        if debug:
            print("Applied: CLAHE equalize (low contrast)")

    if debug:
        print("[Applied steps]:", " -> ".join(steps) if steps else "none")

    return result, steps, damage_mask


#cli
def parse_args():
    p = argparse.ArgumentParser(description="Automatic Image Restoration (RL + fallback)")
    p.add_argument("--input", "-i", required=True, help="Input image path")
    p.add_argument("--output", "-o", default="results/auto_restored.png", help="Output restored image path")
    p.add_argument("--show", action="store_true", help="Show before/after images")
    p.add_argument("--debug", action="store_true", help="Print debug info")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Cannot open image {args.input}")

    ensure_dir_for_file(args.output)
    results_dir = os.path.dirname(args.output) or "results"
    os.makedirs(results_dir, exist_ok=True)

    restored, steps, damage_mask = auto_restore(img, debug=args.debug)

    cv2.imwrite(args.output, restored)
    
    try:
        before = img
        after = restored
        
        h = max(before.shape[0], after.shape[0])
        
        def scale_to_height(im, H):
            h, w = im.shape[:2]
            if h == H:
                return im
            scale = H / h
            return cv2.resize(im, (int(w*scale), H), interpolation=cv2.INTER_AREA)
        b_resized = scale_to_height(before, h)
        a_resized = scale_to_height(after, h)
        side = np.hstack((b_resized, a_resized))
        bvap_path = os.path.join(results_dir, "before_vs_after.png")
        cv2.imwrite(bvap_path, side)
        if args.debug:
            print("Saved before_vs_after image to", bvap_path)
    except Exception as e:
        if args.debug:
            print("Failed to save before_vs_after:", e)

    print("\n✅ Saved restored image →", args.output)
    if args.show:
        show_bgr(img, "Original")
        if damage_mask is not None and damage_mask.any():
            mask_bgr = cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(img, 0.7, mask_bgr, 0.3, 0)
            show_bgr(overlay, "Detected damage (mask overlay)")
        show_bgr(restored, "Auto Restored")

