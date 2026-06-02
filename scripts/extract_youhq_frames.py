#!/usr/bin/env python
# coding=utf-8
"""Extract frames from a tree of video clips (e.g. YouHQ-Train) to PNG folders.

    <src>/<category>/<youtube_id>/<clip>.mp4
        ->  <dst>/<category>__<youtube_id>__<clip>/00000000.png ...

PNG is lossless, so the extracted GT keeps YouHQ's high quality. Full extraction
is huge (~7-11 TB for all of YouHQ-Train), so use --max_clips and/or --every to
extract a manageable subset for fine-tuning.

Example (2000 clips, every frame):
    python scripts/extract_youhq_frames.py \
        --src /mnt/HDD_raid1/yjcho/data/YouHQ-Train \
        --dst /mnt/HDD_raid1/yjcho/data/YouHQ-Train-frames \
        --max_clips 2000 --workers 8 --list_out dataset/youhq_train_list.txt

Requires ffmpeg on PATH.
"""

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob


def extract_one(mp4_path, src_root, dst_root, every, ext):
    rel = os.path.relpath(mp4_path, src_root)
    key = os.path.splitext(rel)[0].replace(os.sep, '__')
    out_dir = os.path.join(dst_root, key)
    os.makedirs(out_dir, exist_ok=True)
    # skip if already extracted
    if any(f.lower().endswith('.png') for f in os.listdir(out_dir)):
        return key, 'skip'
    vf = f'select=not(mod(n\\,{every}))' if every > 1 else None
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', mp4_path]
    if vf:
        cmd += ['-vf', vf, '-vsync', '0']
    cmd += [os.path.join(out_dir, '%08d.png')]
    try:
        subprocess.run(cmd, check=True)
        return key, 'ok'
    except subprocess.CalledProcessError as e:
        return key, f'fail({e.returncode})'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='Root of the .mp4 tree.')
    ap.add_argument('--dst', required=True, help='Output root for frame folders.')
    ap.add_argument('--ext', default='mp4', help='Video extension to scan for.')
    ap.add_argument('--max_clips', type=int, default=None, help='Cap number of clips extracted.')
    ap.add_argument('--every', type=int, default=1, help='Keep every Nth frame (1 = all).')
    ap.add_argument('--workers', type=int, default=8, help='Parallel ffmpeg processes.')
    ap.add_argument('--list_out', default=None, help='Write extracted clip keys to this file.')
    args = ap.parse_args()

    clips = sorted(glob(os.path.join(args.src, '**', f'*.{args.ext}'), recursive=True))
    if args.max_clips:
        clips = clips[:args.max_clips]
    print(f'Found {len(clips)} clips to extract -> {args.dst}')
    os.makedirs(args.dst, exist_ok=True)

    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, c, args.src, args.dst, args.every, args.ext) for c in clips]
        for i, f in enumerate(as_completed(futs), 1):
            key, status = f.result()
            done.append(key)
            if i % 50 == 0 or i == len(clips):
                print(f'  [{i}/{len(clips)}] last={status}')

    if args.list_out:
        with open(args.list_out, 'w') as fout:
            fout.write('\n'.join(sorted(done)) + '\n')
        print(f'Wrote clip list -> {args.list_out}')


if __name__ == '__main__':
    main()
