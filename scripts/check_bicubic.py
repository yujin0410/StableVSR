"""Check which bicubic kernel a dataset's LR was generated with.

A degradation mismatch between training LR and test LR (e.g. MATLAB imresize vs
cv2 INTER_CUBIC -- both "bicubic x4" but different anti-aliasing) is a common
cause of out-of-domain SR collapse. This downsamples a GT frame with several
methods and reports PSNR to the *provided* LR; the method with ~40-99 dB is the
kernel that made it.

Run for your TRAIN LR and for each test set; if they match different methods,
that's your OOD culprit.

    python scripts/check_bicubic.py --gt <GT_root> --lr <LR_root> --scale 4
"""
import os
import argparse
import numpy as np
import cv2

try:
    from basicsr.utils.matlab_functions import imresize as matlab_imresize
except Exception:
    matlab_imresize = None


def first_pair(gt_dir, lr_dir):
    for seq in sorted(os.listdir(gt_dir)):
        g, l = os.path.join(gt_dir, seq), os.path.join(lr_dir, seq)
        if os.path.isdir(g) and os.path.isdir(l):
            gf = sorted(f for f in os.listdir(g) if f.lower().endswith(('.png', '.jpg')))
            lf = sorted(f for f in os.listdir(l) if f.lower().endswith(('.png', '.jpg')))
            if gf and lf:
                return os.path.join(g, gf[0]), os.path.join(l, lf[0])
    return None, None


def psnr(a, b):
    if a.shape != b.shape:
        return -1.0
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--lr', required=True)
    ap.add_argument('--scale', type=int, default=4)
    a = ap.parse_args()

    gp, lp = first_pair(a.gt, a.lr)
    print(f'GT: {gp}\nLR: {lp}')
    gt = cv2.imread(gp)
    lr = cv2.imread(lp)
    print(f'GT shape {gt.shape}  LR shape {lr.shape}')
    h, w = gt.shape[:2]

    # cv2 cubic (no AA) and area
    cv_cubic = cv2.resize(gt, (w // a.scale, h // a.scale), interpolation=cv2.INTER_CUBIC)
    cv_area = cv2.resize(gt, (w // a.scale, h // a.scale), interpolation=cv2.INTER_AREA)
    print(f'  cv2 INTER_CUBIC  vs LR : {psnr(cv_cubic, lr):.2f} dB')
    print(f'  cv2 INTER_AREA   vs LR : {psnr(cv_area, lr):.2f} dB')

    if matlab_imresize is not None:
        ml = matlab_imresize(gt.astype(np.float32) / 255.0, 1.0 / a.scale)
        ml = np.clip(ml * 255.0, 0, 255).round().astype(np.uint8)
        print(f'  MATLAB imresize  vs LR : {psnr(ml, lr):.2f} dB')
    else:
        print('  (basicsr matlab imresize unavailable)')

    print('-> the method with ~40-99 dB is the kernel that generated this LR.')


if __name__ == '__main__':
    main()
