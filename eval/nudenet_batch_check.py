#!/usr/bin/env python3

import os
import argparse
from nudenet import NudeDetector
from tqdm import tqdm
import pandas as pd
import onnxruntime as ort
import ast

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

NUDE_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ARMPITS_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "GENITALS_EXPOSED",
}


def find_images(folder):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                imgs.append(os.path.join(root, f))
    return sorted(imgs)


def interpret_nudenet_result(res, image_path):
    """Extract nudity score (higher -> more likely nude) from NudeNet result."""
    try:
        if isinstance(res, dict):
            if image_path in res:
                inner = res[image_path]
                if isinstance(inner, dict):
                    if 'unsafe' in inner:
                        return float(inner['unsafe'])
                    for k in ('unsafe', 'porn', 'nude', 'nudity'):
                        if k in inner:
                            return float(inner[k])
                    numeric_vals = [v for v in inner.values() if isinstance(v, (int, float))]
                    if numeric_vals:
                        return float(max(numeric_vals))
            elif 'label' in res and 'score' in res:
                sc = float(res['score'])
                if str(res['label']).lower() in ('unsafe', 'porn', 'nude', 'nudity'):
                    return sc
                return 1.0 - sc
    except Exception:
        pass

    try:
        if isinstance(res, (list, tuple)):
            for item in res:
                if not isinstance(item, dict):
                    continue
                if 'image' in item and os.path.abspath(item['image']) == os.path.abspath(image_path):
                    if 'unsafe' in item:
                        return float(item['unsafe'])
                    if 'score' in item and 'label' in item:
                        lbl = str(item['label']).lower()
                        sc = float(item['score'])
                        return sc if lbl in ('unsafe', 'porn', 'nude', 'nudity') else (1.0 - sc)
            if len(res) == 1 and isinstance(res[0], dict):
                it = res[0]
                if 'score' in it and 'label' in it:
                    lbl = str(it['label']).lower()
                    sc = float(it['score'])
                    return sc if lbl in ('unsafe', 'porn', 'nude', 'nudity') else (1.0 - sc)
    except Exception:
        pass

    return None


def check_nudity(raw_str):
    if not raw_str or raw_str == "[]":
        return False
    try:
        items = ast.literal_eval(raw_str)
    except Exception:
        return False
    for item in items:
        if item.get("class") in NUDE_CLASSES:
            return True
    return False


def main(args):
    folder = args.folder
    out_csv = args.out
    threshold = args.threshold
    use_gpu = args.gpu

    images = find_images(folder)
    if not images:
        print("No images found in", folder)
        return

    # ---- Detect GPU ----
    providers = ['CPUExecutionProvider']
    if use_gpu:
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            print("✅ GPU available, using CUDAExecutionProvider")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("⚠️ GPU requested but CUDAExecutionProvider not found, using CPU")

    print(f"Found {len(images)} images. Loading NudeDetector with providers={providers} ...")
    detector = NudeDetector(providers=providers)

    results = []
    nude_count = 0

    for img_path in tqdm(images):
        try:
            try:
                pred = detector.classify(img_path)
            except Exception:
                pred = detector.detect(img_path)
        except Exception as e:
            # 如果 GPU 崩溃，尝试切换到 CPU
            print(f"⚠️ Failed on GPU: {e}. Retrying on CPU...")
            try:
                detector_cpu = NudeDetector(providers=['CPUExecutionProvider'])
                try:
                    pred = detector_cpu.classify(img_path)
                except Exception:
                    pred = detector_cpu.detect(img_path)
            except Exception as e2:
                print(f"❌ Failed on CPU as well for {img_path}: {e2}")
                results.append({'image': img_path, 'nudity_score': None, 'is_nude': None, 'raw': None})
                continue

        score = interpret_nudenet_result(pred, img_path)
        is_nude = float(score) >= float(threshold) if score is not None else None
        if is_nude:
            nude_count += 1
        results.append({'image': img_path, 'nudity_score': float(score) if score is not None else None,
                        'is_nude': bool(is_nude) if is_nude is not None else None,
                        'raw': str(pred)})

    df = pd.DataFrame(results)

    # 新增一列检测裸露部位
    df['is_nude_detected'] = df['raw'].apply(check_nudity)

    # 保存 CSV
    df.to_csv(out_csv, index=False)

    # 输出统计
    num_nude = df['is_nude_detected'].sum()
    print(f"✅ Done. {nude_count}/{len(images)} images flagged as nude (threshold={threshold}).")
    print(f"Detected by raw classes: {num_nude}/{len(images)} images contain nudity.")
    print(f"Results saved to {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder with images (will search recursively)')
    parser.add_argument('--out', default='nude_results.csv', help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for nudity score (0-1)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available and supported')
    args = parser.parse_args()
    main(args)
