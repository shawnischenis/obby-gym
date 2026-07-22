#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REAR_VIDEO SIDE_VIDEO OUTPUT_MP4" >&2
  exit 2
fi

ffmpeg -i "$1" -i "$2" \
  -filter_complex \
  "[0:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2[rear];[1:v]setpts=PTS-STARTPTS,scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2[side];[rear][side]hstack=inputs=2:shortest=1[out]" \
  -map "[out]" -an -r 30 -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p \
  "$3"
