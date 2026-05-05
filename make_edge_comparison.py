"""Edge map comparison via Sobel filter.

For one chosen frame, apply a Sobel edge detector to each method's
output and visualize side-by-side. Sharper, more accurate edges
indicate better detail preservation.

Usage:
    python make_edge_comparison.py --seq 000 --frame 50 \
        --crop 600 350 200 200 --output figures/edge_comparison.pdf
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import sobel


PATHS = {
    'LR (bicubic up)': '/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic/{seq}/{idx:08d}.png',
    'BasicVSR++':      '/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/reds/{seq}/{idx:08d}.png',
    'StableVSR':       '/mnt/HDD_raid1/yjcho/stablevsr_reds/{seq}/{idx:08d}.png',
    'Ours':            '/mnt/HDD_raid1/yjcho/20260430/reds/{seq}/{idx:08d}.png',
    'GT':              '/mnt/HDD_raid1/yjcho/data/REDS/test/gt/{seq}/{idx:08d}.png',
}


def load_y(path, upscale4=False):
    img = Image.open(path).convert('RGB')
    if upscale4:
        img = img.resize((img.width * 4, img.height * 4), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def sobel_magnitude(y):
    gx = sobel(y, axis=1)
    gy = sobel(y, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', default='000')
    p.add_argument('--frame', type=int, default=50)
    p.add_argument('--crop', type=int, nargs=4, default=[600, 350, 200, 200],
                   metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--output', default='figures/edge_comparison.pdf')
    args = p.parse_args()

    x, y, w, h = args.crop
    methods = list(PATHS.keys())
    n = len(methods)

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))

    for col, name in enumerate(methods):
        path = PATHS[name].format(seq=args.seq, idx=args.frame)
        if not os.path.isfile(path):
            print(f"Missing: {path}")
            for r in range(2):
                axes[r, col].axis('off')
            continue
        upscale4 = (name == 'LR (bicubic up)')
        y_img = load_y(path, upscale4=upscale4)
        H, W = y_img.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        patch = y_img[y1:y2, x1:x2]
        edge = sobel_magnitude(patch)

        # row 0: grayscale patch
        axes[0, col].imshow(patch, cmap='gray')
        axes[0, col].set_title(name, fontsize=13)
        axes[0, col].axis('off')

        # row 1: sobel magnitude
        axes[1, col].imshow(edge, cmap='magma')
        axes[1, col].axis('off')
        if col == 0:
            axes[0, col].text(-0.08, 0.5, 'Image',
                              transform=axes[0, col].transAxes,
                              fontsize=12, va='center', ha='right', fontweight='bold')
            axes[1, col].text(-0.08, 0.5, 'Sobel',
                              transform=axes[1, col].transAxes,
                              fontsize=12, va='center', ha='right', fontweight='bold')

    plt.tight_layout(pad=0.5, h_pad=0.3, w_pad=0.3)
    out_dir = os.path.dirname(args.output) or '.'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(args.output)[0] + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
