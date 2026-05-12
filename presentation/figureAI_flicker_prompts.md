# figureAI 프롬프트 — Motivation 슬라이드 보강용

내가 만든 `motivation_flicker.png` 가 마음에 안 든다면 아래 프롬프트로 figureAI에 맡겨주세요. 두 가지 변형을 준비했어요.

---

## A. 자연 이미지 기반 (가장 직관적)

**용도**: 실제 영상 patch를 사용해서 "같은 장면인데 프레임마다 texture가 다르게 합성되는" 현상을 보여주는 그림.

**English prompt**

> A clean academic figure with two horizontal rows of three small image patches each. Top row labeled "Per-frame independent stochastic denoising" in red. Each of the three patches shows a zoomed-in close-up of human hair or fabric texture — the underlying scene is the same in all three, but the fine high-frequency strands differ subtly between frames as if randomly resampled. A red dashed border around all three patches and small red wavy arrows between them, plus the text "different HF detail → flicker" on the right.
>
> Bottom row labeled "Frequency-conditioned (WC-BD-SFT)" in blue. Three patches with identical fine-detail hair/fabric texture — the strands match across frames. Blue border, blue arrows, and the text "consistent HF detail → stable" on the right.
>
> Photorealistic close-up texture, side-by-side comparison, minimal academic style, white background, navy/blue/red accent colors.

**한국어**

> 학술 비교용 시각화. 상하 두 행, 각 행에 3개의 영상 patch.
> 윗줄(빨강, "Per-frame independent stochastic denoising") — 머리카락 또는 직물의 클로즈업 texture. 세 patch는 같은 장면이지만 세부 high-frequency 가닥이 프레임마다 미세하게 다름. 빨강 dashed border + 빨강 wave 화살표. 오른쪽에 "different HF detail → flicker".
> 아랫줄(파랑, "Frequency-conditioned (WC-BD-SFT)") — 동일한 fine detail이 3 프레임 모두에 일관되게 유지됨. 파랑 border + 파랑 화살표 + "consistent HF detail → stable".
> Photorealistic close-up, 흰 배경, navy/blue/red accent.

---

## B. 단순한 도식 (vector style, 더 깔끔)

**English prompt**

> An academic vector illustration showing texture inconsistency between video frames. Top row: three rectangular frames side-by-side, each filled with a similar wavy gray pattern but the fine details inside differ between frames. The frames have red outlines, with the label "Stochastic per-frame denoising" above them in red bold text. Add small red curly arrows between frames suggesting flicker. To the right write "different fine detail ⇒ flicker".
>
> Bottom row: three rectangular frames, all filled with the EXACT SAME wavy gray pattern (consistent fine detail). Blue outlines, label "Frequency-conditioned" in blue bold. To the right write "stable detail ⇒ no flicker".
>
> Flat vector design, no photorealism, navy text on white background, only red and blue as accent colors.

**한국어**

> 영상 프레임 간 texture 불일치를 표현한 학술용 vector 일러스트.
> 윗줄 — 세 개의 직사각형 frame, 비슷한 회색 wavy 패턴인데 fine detail이 프레임마다 다름. 빨강 outline, 위에 "Stochastic per-frame denoising" 빨강 볼드. 프레임 사이에 빨강 curly 화살표. 오른쪽에 "different fine detail ⇒ flicker".
> 아랫줄 — 세 직사각형 frame, 정확히 같은 패턴(일관된 fine detail). 파랑 outline, "Frequency-conditioned" 파랑 볼드. 오른쪽 "stable detail ⇒ no flicker".
> Flat vector, photorealism 금지, 흰 배경, 빨강/파랑만 accent.

---

## C. 실험 결과 기반 (만약 본인의 실제 결과 영상이 있다면)

본인의 StableVSR baseline output 비디오와 WC-BD-SFT output 비디오에서 같은 영역의 patch를 3 프레임씩 잘라낸 뒤 위/아래로 비교하는 게 가장 강력합니다. 이건 어떤 AI도 못 합니다 — 실제 inference 결과로만 만들 수 있어요.

스크립트로 만든다면:
```python
# 두 비디오 폴더에서 patch 추출
import numpy as np, cv2
for k, frame_idx in enumerate([45, 46, 47]):
    img_base = cv2.imread(f"out_stablevsr/frame_{frame_idx:04d}.png")
    crop_base = img_base[200:300, 400:500]  # 텍스처가 풍부한 영역 선택
    cv2.imwrite(f"patch_baseline_{k}.png", crop_base)

    img_ours = cv2.imread(f"out_wcbdsft/frame_{frame_idx:04d}.png")
    crop_ours = img_ours[200:300, 400:500]
    cv2.imwrite(f"patch_ours_{k}.png", crop_ours)
```
그 다음 PowerPoint에서 6개 patch를 2×3 grid로 배치하면 됩니다.

---

## 활용 팁

- A 가 가장 청중에게 와닿지만 LLM이 hair/fabric texture를 일관되게 생성하기 어려울 수 있어요. 여러 번 시도해서 골라 쓰세요.
- B 는 LLM이 더 일관되게 만들 수 있는 안전한 옵션입니다.
- C 가 가능하다면 가장 설득력 있습니다 — 본인의 실제 결과니까요.
- 만들어진 이미지에 글자가 깨져있으면 PPT에서 textbox로 덧입히세요.
