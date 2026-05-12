# figureAI 프롬프트 — 슬라이드 보강용 다이어그램

발표 슬라이드의 시각 자료를 더 보강하고 싶을 때 사용할 수 있는 프롬프트들입니다. 본문 풀어 쓰기보다 "diagram-style", "flat vector", "academic poster style"이라는 표현을 함께 주는 게 효과가 좋습니다.

각 프롬프트는 영문 / 한국어 두 버전으로 제공합니다.

---

## 1. Temporal Flickering 시각화 (Slide 3 · Motivation)

**용도**: 프레임마다 stochastic하게 합성된 high-frequency detail이 어떻게 frame 간 inconsistent하게 보이는지를 직관적으로 보여주는 다이어그램.

**English prompt**

> A clean, academic-style diagram showing video frames in a horizontal row, t-1, t, t+1, t+2. Each frame shows a zoomed-in patch of a textured surface (e.g., fabric or hair). The patches across frames are slightly different in texture pattern, with small red wiggly arrows between frames to indicate "flickering". On the right, the same row but with stable texture (no red arrows). Minimal, flat vector illustration style, navy and coral color palette, white background, no photorealism.

**한국어 프롬프트**

> 시간순으로 t-1, t, t+1, t+2의 비디오 프레임을 가로로 늘어놓고, 각 프레임에서 texture가 확대된 patch를 보여주는 academic-style 다이어그램. 왼쪽 행은 frame 간 texture pattern이 조금씩 다르게 그려지고, 작은 빨간 wave 화살표로 "flickering"을 표시. 오른쪽 행은 동일한 frame들이지만 texture가 안정적으로 일관되게 표현됨. Flat vector 일러스트, navy + coral 색감, 흰 배경, photorealistic 금지.

---

## 2. Perception–Distortion Trade-off Curve (Slide 18 · Metrics)

**용도**: PSNR(distortion)과 LPIPS/MUSIQ(perception)이 trade-off 관계임을 곡선으로 시각화. 각 방법(BasicVSR++, StableVSR, DGAF-VSR, Ours)을 곡선 위 점으로 표시.

**English prompt**

> A scientific 2D scatter / curve plot. X-axis labeled "Distortion (lower → higher fidelity)" decreasing to the right; Y-axis labeled "Perceptual quality (higher = better)". A smooth gray dashed curve from upper-left to lower-right represents the perception–distortion trade-off frontier. Four labeled points on the curve: BasicVSR++ (lower-right region: low perception, high fidelity, gray dot), StableVSR (middle: blue dot), DGAF-VSR (slightly above StableVSR: teal dot), WC-BD-SFT/Ours (upper-left: high perception, lower fidelity, red star). Clean academic style, white background, axes in dark navy, light gridlines. Title at top: "Perception–Distortion Trade-off".

**한국어 프롬프트**

> Distortion(낮을수록 좋음, X축)과 Perceptual quality(높을수록 좋음, Y축)의 trade-off curve를 보여주는 학술 plot. 회색 dashed curve가 좌상단에서 우하단으로 부드럽게 이어지고, 곡선 위에 네 점이 배치 — BasicVSR++ (우하단, 회색), StableVSR (중간, 파랑), DGAF-VSR (중간 위, 청록), Ours (좌상단, 빨강 별표). 흰 배경, navy 축선, 옅은 grid. 제목 "Perception–Distortion Trade-off".

---

## 3. Bidirectional Sampling Schematic (Slide 7 · Preliminaries)

**용도**: StableVSR의 Frame-wise Bidirectional Sampling을 시각화. 시간축에서 forward/backward를 번갈아 진행하는 모습.

**English prompt**

> A flat schematic showing T frames laid horizontally as small image-like rectangles. Above the row, a curved arrow loops over the frames in a zig-zag pattern: forward arrow from left to right at step 1, then a reverse arrow from right to left at step 2, then forward again. Each arrow is labeled "step 1", "step 2", "step 3". The arrows are alternating blue and red. Title: "Frame-wise Bidirectional Sampling". Minimal academic style, navy text on white background.

**한국어 프롬프트**

> T개의 프레임이 가로로 늘어선 위쪽에서, 화살표가 zig-zag 패턴으로 흐르는 schematic. step 1에서는 왼→오 (forward), step 2에서는 오→왼 (backward), step 3에서는 다시 왼→오. 화살표는 파랑/빨강 교대. 흰 배경, navy 텍스트, 미니멀 academic style. 제목 "Frame-wise Bidirectional Sampling".

---

## 4. WCM Block Diagram (Slide 10 · Architecture로 대체용)

**용도**: 본인의 thesis Fig 3.1을 더 깔끔한 vector style로 다시 그리고 싶을 때.

**English prompt**

> An academic block diagram of the Wavelet Conditioning Module (WCM). Top-left: input image labeled "I^LR_t". An arrow leads to a block labeled "DT-CWT (J=4)". From there, four parallel paths labeled "L=1", "L=2", "L=3", "L=4" each go into identical rounded rectangles labeled "SubbandBlock". The L=1 and L=2 paths merge into a blue arrow labeled "γ_H, β_H → up_blocks[1] (decoder)". The L=3 and L=4 paths merge into a red arrow labeled "γ_L, β_L → down_blocks[1] (encoder)". On the right, a simplified U-Net silhouette with the two injection points highlighted. Clean flat design, flat colors (navy, blue, red, gray), no shadows.

**한국어 프롬프트**

> Wavelet Conditioning Module (WCM)의 블록 다이어그램. 좌측 상단에서 입력 영상 "I^LR_t"가 화살표를 따라 "DT-CWT (J=4)" 블록으로 진입. 거기서 L=1, L=2, L=3, L=4 네 갈래가 각각 동일한 "SubbandBlock" 둥근 사각형 블록으로 들어감. L=1, L=2 출력은 파란색 화살표 "γ_H, β_H → up_blocks[1] (decoder)"로 합쳐지고, L=3, L=4 출력은 빨강 화살표 "γ_L, β_L → down_blocks[1] (encoder)"로 합쳐짐. 오른쪽에는 simplified U-Net silhouette을 그리고 두 injection point를 하이라이트. Flat design, navy/blue/red/gray, 그림자 없음.

---

## 5. Wavelet Coefficient Recombination 비유 (Slide 14)

**용도**: Magnitude · cos · sin 결합이 polar-to-Cartesian과 같음을 직관적으로 보여주는 보조 그림.

**English prompt**

> A two-panel side-by-side educational figure.
> Left panel: a complex plane (Argand diagram) with a vector from origin labeled "C = R·e^(iφ)". The vector's length is labeled R, the angle from real axis is labeled φ. The horizontal projection is labeled "R cos φ", vertical "R sin φ". Clean labels, grid.
> Right panel: the same concept but in "feature space" — three rounded rectangles labeled "e_M", "e_cos", "e_sin" with element-wise multiplication symbols ⊙ between them, producing "e_re = e_M ⊙ e_cos" and "e_im = e_M ⊙ e_sin". A subtitle below reads: "Polar-to-Cartesian, mirrored in the embedding space".
> Academic style, navy / purple palette, white background.

**한국어 프롬프트**

> 두 패널 좌우 비교 일러스트.
> 왼쪽 — Argand diagram. 원점에서 출발하는 벡터 "C = R·e^(iφ)", 길이는 R, 실수축과 이루는 각도는 φ. 수평 projection은 "R cos φ", 수직은 "R sin φ"로 label.
> 오른쪽 — 동일한 개념을 feature space에서 재현. "e_M", "e_cos", "e_sin" 세 개의 둥근 사각형이 element-wise multiplication ⊙로 결합되어 "e_re = e_M ⊙ e_cos", "e_im = e_M ⊙ e_sin" 산출. 하단 부제 "Polar-to-Cartesian, mirrored in the embedding space".
> Navy / purple 팔레트, 흰 배경, academic style.

---

## 활용 팁

- 모든 프롬프트에 `flat vector`, `academic poster style`, `minimal`, `no photorealism`을 추가하면 슬라이드 톤과 잘 어울립니다.
- 결과 이미지가 너무 화려하면 `monochrome with one accent color` 또는 `navy and white only` 같은 색 제약을 강조해 보세요.
- 글자가 들어가야 하는 다이어그램은 LLM이 종종 글자를 깨뜨리기 때문에, 텍스트는 PPT에서 직접 입히는 게 안전합니다 — 다이어그램만 받고 라벨은 PowerPoint에서 textbox로 덧붙이세요.
