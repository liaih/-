#!/usr/bin/env bash
# 室內設計短片草稿：6 張施工現場照片 → 9:16 直式約 20 秒，含 CHEN YANG 浮水印
# 用法：bash shortfilm/make_short.sh [輸出檔]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-shortfilm/chenyang_short_draft.mp4}"
FONT=/usr/share/fonts/truetype/freefont/FreeSerif.ttf
W=1080; H=1920; FPS=30; D=3.6; X=0.6   # 每鏡秒數、轉場秒數
N=$(awk "BEGIN{print int($D*$FPS)}")
ORDER=(4 6 2 5 7 1)                    # 走廊大景 → 各房間牆面 → 收尾
# 每鏡運鏡：in=推近、l2r/r2l=橫移
MOVE=(in l2r in r2l l2r in)

inputs=(); filters=""
for i in "${!ORDER[@]}"; do
  inputs+=(-loop 1 -t "$D" -i "${ORDER[$i]}.jpg")
  case ${MOVE[$i]} in
    in)  z="min(1+0.12*on/$N,1.12)"; x="iw/2-(iw/zoom/2)"; y="ih/2-(ih/zoom/2)";;
    l2r) z="1.0"; x="(iw-iw/zoom)*on/$N"; y="0";;
    r2l) z="1.0"; x="(iw-iw/zoom)*(1-on/$N)"; y="0";;
  esac
  # 先裁成 9:16 的較寬區域供橫移，再放大避免抖動
  filters+="[$i:v]scale=-2:2400,crop='min(iw,ih*0.75)':ih,setsar=1,"
  filters+="zoompan=z='$z':x='$x':y='$y':d=$N:s=${W}x${H}:fps=$FPS,"
  filters+="eq=brightness=0.03:saturation=0.9,format=yuv420p[v$i];"
done
# 交叉溶接
prev="v0"; off=0
for i in $(seq 1 $((${#ORDER[@]}-1))); do
  off=$(awk "BEGIN{print $off+$D-$X}")
  filters+="[$prev][v$i]xfade=transition=fade:duration=$X:offset=$off[x$i];"
  prev="x$i"
done
total=$(awk "BEGIN{print $off+$D}")
# 結尾淡出白 + 浮水印
filters+="[$prev]tpad=stop_mode=clone:stop_duration=1.4,fade=t=out:st=$(awk "BEGIN{print $total-0.2}"):d=1.2:color=white,"
filters+="drawtext=fontfile=$FONT:text='C H E N   Y A N G':fontsize=44:fontcolor=white@0.85:x=(w-tw)/2:y=h*0.86:shadowcolor=black@0.25:shadowx=1:shadowy=1,"
filters+="drawtext=fontfile=$FONT:text='INTERIOR  DESIGN':fontsize=20:fontcolor=white@0.75:x=(w-tw)/2:y=h*0.86+58,"
filters+="drawtext=fontfile=$FONT:text='C H E N   Y A N G':fontsize=64:fontcolor=0x8a7a63:x=(w-tw)/2:y=h/2-40:enable='gte(t,$total+0.4)',"
filters+="drawtext=fontfile=$FONT:text='INTERIOR  DESIGN':fontsize=28:fontcolor=0x8a7a63:x=(w-tw)/2:y=h/2+50:enable='gte(t,$total+0.4)'[out]"

ffmpeg -y -v error "${inputs[@]}" -filter_complex "$filters" -map "[out]" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUT"
echo "done: $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")s)"
