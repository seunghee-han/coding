# -*- coding: utf-8 -*-
"""
WSL 전용: PDF -> (필요시) OCR(EasyOCR 우선, Tesseract 폴백) -> 이미지/캡션 -> CLIP 매칭
-> HTML 뷰어 생성 -> 로컬 서버 자동 오픈(WSL은 wslview로 윈도우 브라우저까지 띄움)
"""
from __future__ import annotations
from pathlib import Path
import json, re, threading, socket, time, subprocess, sys, platform, shutil
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ===================== 1) 설정 =====================
PDF_PATH = Path("./HOJBC0_2016_v20n9_1649.pdf")

def _pick_pdf_from_argv(default_path: Path) -> Path:
    # Jupyter/VSCode가 넣는 --f=... 같은 옵션은 무시
    for a in sys.argv[1:]:
        if a.startswith("-"):
            continue
        # file:// 형태면 정리
        if a.startswith("file://"):
            a = a[7:]
        p = Path(a)
        if p.suffix.lower() == ".pdf" and p.exists():
            return p
    return default_path

PDF_PATH = _pick_pdf_from_argv(PDF_PATH)

OUT_ROOT = Path("./result")
DOC_NAME = PDF_PATH.stem
OUT_DIR  = OUT_ROOT / DOC_NAME

PAGES_DIR  = OUT_DIR / "pages"
IMAGES_DIR = OUT_DIR / "images"
MANIFEST   = OUT_DIR / "manifest.json"
INDEX_JSON = OUT_DIR / "index.json"
HTML_OUT   = OUT_DIR / "viewer_clip.html"

for d in [OUT_DIR, PAGES_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 매칭/렌더
CONTEXT_RADIUS_PX = 0
CONTEXT_MAX_WORDS = 0
SIM_THRESHOLD = 0.20
TOP_K = 1
MIN_WORD_LEN = 2
PNG_SCALE = 2.0

# OCR
USE_OCR_ALWAYS = True
OCR_MIN_WORDS_FOR_SKIP = 5
OCR_LANG = "ko"  # EasyOCR 기준 (한국어+영어는 자동 혼용)

# 캡션 탐지/합성
CAPTION_SEARCH_HEIGHT   = 10
CAPTION_SIDE_OVERLAP    = 0.10
CAPTION_MAX_WIDTH_RATIO = 0.90
CAPTION_CENTER_TOL      = 40
CAPTION_MIN_TOKENS      = 1
CAPTION_MAX_VERTICAL_GAP = 10
CAPTION_HEAD_RE = re.compile(r'^(fig(?:\.|ure)?|그림|표|table|도)\s*[\.:]?\s*\d+', re.IGNORECASE)

ALWAYS_ADD_PLACEHOLDER_IF_EMPTY = True
SAVE_WITH_SUFFIX = True  # *_cap.png로 저장

# 캡션 스타일
CAPTION_WRAP_PX_RATIO = 0.92
CAPTION_PADDING       = 12
CAPTION_LINE_SPACING  = 6
CAPTION_BG            = (255, 255, 255)
CAPTION_FG            = (0, 0, 0)
CAPTION_BORDER        = (220, 220, 220)
CAPTION_BORDER_W      = 1

# 임베딩 융합(이미지+캡션)
FUSE_WITH_CAPTION = True
FUSE_IMG_WEIGHT   = 0.40

# CLIP 모델 (가볍고 호환 잘 되는 기본형)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# ===================== 2) 유틸 =====================
def _to_jsonable(obj):
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            import base64 as _b64
            return "base64:" + _b64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and obj.__class__.__name__ == "Rect":
        return [float(obj.x0), float(obj.y0), float(obj.x1), float(obj.y1)]
    return obj

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)

def _overlap_ratio(ax0, ax1, bx0, bx1):
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    aw = max(1e-6, ax1 - ax0)
    bw = max(1e-6, bx1 - bx0)
    return inter / min(aw, bw)

def _px2pt(bbox_px, scale: float):
    x0,y0,x1,y1 = bbox_px
    return [x0/scale, y0/scale, x1/scale, y1/scale]

# ===================== 3) OCR (EasyOCR 우선, Tesseract 폴백) =====================
_EASY = None
def _easy_reader():
    global _EASY
    if _EASY is None:
        import easyocr
        langs = ["ko","en"] if OCR_LANG.startswith(("ko","kor")) else ["en"]
        model_dir = (OUT_DIR / ".easyocr").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            pass
        _EASY = easyocr.Reader(
            langs, gpu=use_gpu,
            verbose=False, model_storage_directory=str(model_dir),
            download_enabled=True
        )
        # 웜업
        try:
            tmp = OUT_DIR / "_warmup.png"
            if not tmp.exists():
                Image.new("RGB", (32, 16), (255,255,255)).save(tmp)
            _EASY.readtext(str(tmp))
        except Exception:
            pass
    return _EASY

def run_ocr_on_png(png_path: Path):
    words = []
    try:
        reader = _easy_reader()
        res = reader.readtext(str(png_path), detail=1)  # [(bbox, text, conf), ...]
        for bbox, txt, conf in res:
            if not txt or not txt.strip():
                continue
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            x0, y0, x1, y1 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            words.append({"text": txt.strip(), "bbox_px": [x0,y0,x1,y1], "conf": float(conf)})
    except Exception as e:
        print("[OCR] EasyOCR 실패:", e)
        # (옵션) pytesseract 폴백: 설치돼 있을 때만
        try:
            import pytesseract
            img = Image.open(png_path).convert("RGB")
            data = pytesseract.image_to_data(img, lang="kor+eng", output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                if not txt: continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                conf = float(data.get("conf", [0]*n)[i] if data.get("conf") else 0)
                words.append({"text": txt, "bbox_px": [float(x), float(y), float(x+w), float(y+h)], "conf": conf})
        except Exception as e2:
            print("[OCR] Tesseract 폴백도 실패:", e2)
    return words

# ===================== 4) PDF 파싱 (OCR 통합) =====================
def parse_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    meta = {"page_count": len(doc), "pages": []}

    for pno, page in enumerate(doc):
        width, height = page.rect.width, page.rect.height

        words_pdf = [{"text": w[4], "bbox": [w[0], w[1], w[2], w[3]]}
                     for w in page.get_text("words")]
        plain_pdf = page.get_text("text")

        image_items = []
        page_xrefs = [info[0] for info in page.get_images(full=True)]
        seen = set()
        for xref in page_xrefs:
            rects = []
            try:
                rects = list(page.get_image_rects(xref))
            except Exception:
                rects = []
            if not rects:
                try:
                    r = page.get_image_bbox(xref)
                    if r is not None:
                        rects = [r]
                except Exception:
                    rects = []
            for r in rects:
                bb = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
                key = (xref, round(bb[0],2), round(bb[1],2), round(bb[2],2), round(bb[3],2))
                if key in seen: continue
                seen.add(key)
                image_items.append({"xref": xref, "bbox": bb})

        pix = page.get_pixmap(matrix=fitz.Matrix(PNG_SCALE, PNG_SCALE))
        png_path = PAGES_DIR / f"page_{pno+1:04d}.png"
        pix.save(str(png_path))

        need_ocr = USE_OCR_ALWAYS or (len(words_pdf) < OCR_MIN_WORDS_FOR_SKIP)
        words_final = words_pdf
        plain_final = plain_pdf

        if need_ocr:
            ocr_words = run_ocr_on_png(png_path)
            if ocr_words:
                tmp = []
                for ow in ocr_words:
                    pt = _px2pt(ow["bbox_px"], PNG_SCALE)
                    tmp.append({"text": ow["text"], "bbox": pt, "conf": ow.get("conf", 0.0)})
                words_final = sorted(tmp, key=lambda w: ((w["bbox"][1]+w["bbox"][3])/2.0, w["bbox"][0]))
                plain_final = " ".join([w["text"] for w in words_final])

        txt_path = PAGES_DIR / f"page_{pno+1:04d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_final or "")

        meta["pages"].append({
            "page": pno, "width": width, "height": height,
            "words": words_final, "images": image_items,
            "png": str(png_path), "used_ocr": bool(need_ocr and len(words_final) > 0)
        })

    doc.close()
    return meta

# ===================== 5) 원본 이미지 추출 =====================
def export_images(pdf_path: Path):
    doc = fitz.open(pdf_path)
    xref_to_path = {}
    for _pno, page in enumerate(doc):
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in xref_to_path: continue
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width <= 220 or pix.height <= 220:
                continue
            img_path = IMAGES_DIR / f"img_{xref}.png"
            with open(img_path, "wb") as f:
                f.write(pix.tobytes("png"))
            xref_to_path[xref] = str(img_path)
    doc.close()
    return xref_to_path

# ===================== 6) 캡션 탐지/합성 =====================
def _lines_with_bbox(words, y_tol=3.5):
    lines = []
    for w in sorted(words, key=lambda x: ((x["bbox"][1]+x["bbox"][3])/2.0, x["bbox"][0])):
        x0,y0,x1,y1 = w["bbox"]; cy = (y0+y1)/2.0
        placed = False
        for L in lines:
            if abs(L["_cy"] - cy) <= y_tol:
                L["items"].append(w)
                L["_cy"] = (L["_cy"]*L["_n"] + cy)/(L["_n"]+1); L["_n"] += 1
                placed = True; break
        if not placed:
            lines.append({"_cy": cy, "_n":1, "items":[w]})
    out=[]
    for L in sorted(lines, key=lambda z: z["_cy"]):
        its = sorted(L["items"], key=lambda it: it["bbox"][0])
        text = " ".join([(it["text"] or "").strip() for it in its if (it.get("text") or "").strip()])
        x0 = min(it["bbox"][0] for it in its); y0 = min(it["bbox"][1] for it in its)
        x1 = max(it["bbox"][2] for it in its); y1 = max(it["bbox"][3] for it in its)
        out.append({"text": text, "bbox":[x0,y0,x1,y1]})
    return out

def extract_caption_for_image(page_dict: dict, img_bbox):
    x0, y0, x1, y1 = img_bbox
    img_cx = (x0 + x1) / 2.0
    img_w  = max(1e-6, x1 - x0)

    below, above = [], []
    for w in page_dict.get("words", []):
        wb = w["bbox"]; t = (w.get("text") or "").strip()
        if not t: continue
        if y1 <= wb[1] <= y1 + CAPTION_SEARCH_HEIGHT: below.append(w)
        if y0 - CAPTION_SEARCH_HEIGHT <= wb[3] <= y0: above.append(w)

    def pick(lines, anchor, direction):
        if not lines: return ""
        cand=[]
        for ln in lines:
            lx0,ly0,lx1,ly1 = ln["bbox"]
            lw=max(1e-6, lx1-lx0); lcx=(lx0+lx1)/2.0
            overlap=_overlap_ratio(x0,x1,lx0,lx1)
            center_ok=abs(lcx-img_cx)<=CAPTION_CENTER_TOL
            width_ok=(lw/img_w) <= CAPTION_MAX_WIDTH_RATIO
            vgap = (ly0 - y1) if direction=='below' else (y0 - ly1)
            if vgap < 0: vgap = 0
            vertical_ok = vgap <= CAPTION_MAX_VERTICAL_GAP
            if (overlap>=CAPTION_SIDE_OVERLAP or center_ok or width_ok) and vertical_ok:
                cand.append(ln)
        if not cand: return ""
        for ln in cand:
            if CAPTION_HEAD_RE.match(ln["text"].strip()): return ln["text"].strip()
        cand.sort(key=lambda ln: abs(((ln["bbox"][1]+ln["bbox"][3])/2.0) - anchor))
        return cand[0]["text"].strip()

    cap = pick(_lines_with_bbox(below), y1, 'below') or pick(_lines_with_bbox(above), y0, 'above')
    return cap if len(cap.split()) >= CAPTION_MIN_TOKENS else ""

def _wrap_by_width(draw, text, font, max_px):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        bb = draw.textbbox((0,0), test, font=font)
        if (bb[2]-bb[0]) <= max_px or not cur: cur = test
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def draw_caption_on_image(img_path: Path, caption: str, placeholder: str | None = None):
    abs_path = Path(img_path).resolve()
    try:
        im = Image.open(abs_path).convert("RGB")
    except Exception as e:
        print(f"[warn] open failed: {abs_path} :: {e}")
        return False
    text = caption if caption else (placeholder or "")
    if not text: return False

    W, H = im.size
    try: font = ImageFont.load_default()
    except Exception: font = ImageFont.load_default()

    PAD = CAPTION_PADDING; LSP = CAPTION_LINE_SPACING
    BG, BD, BDW, FG = CAPTION_BG, CAPTION_BORDER, CAPTION_BORDER_W, CAPTION_FG

    wrap_width_px = int(W * CAPTION_WRAP_PX_RATIO)
    draw_tmp = ImageDraw.Draw(im)
    lines = _wrap_by_width(draw_tmp, text, font, wrap_width_px)

    line_h = []
    for ln in lines:
        bb = draw_tmp.textbbox((0,0), ln, font=font)
        line_h.append(bb[3]-bb[1])
    text_h = sum(line_h) + LSP*(len(lines)-1)
    box_h  = text_h + PAD*2

    new_im = Image.new("RGB", (W, H + box_h), BG)
    new_im.paste(im, (0,0))
    draw = ImageDraw.Draw(new_im)
    draw.rectangle([0, H, W-1, H + box_h - 1], outline=BD, width=BDW)

    y = H + PAD
    for i, ln in enumerate(lines):
        tw = draw.textbbox((0,0), ln, font=font)[2]
        x  = (W - tw)//2
        draw.text((x, y), ln, fill=FG, font=font)
        y += line_h[i] + LSP

    out_path = abs_path.with_name(abs_path.stem + "_cap" + abs_path.suffix) if SAVE_WITH_SUFFIX else abs_path
    new_im.save(out_path)
    print(f"[cap] saved: {out_path}")
    return True

# ===================== 7) CLIP 임베딩 =====================
import torch
from transformers import CLIPModel, CLIPProcessor
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device).eval()
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

@torch.no_grad()
def clip_text_embed(texts):
    if isinstance(texts, str): texts=[texts]
    if not texts: return np.zeros((0, 512), dtype=np.float32)
    batch = clip_proc(text=texts, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(_device) for k, v in batch.items()}
    feats = clip_model.get_text_features(**batch)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.detach().cpu().numpy().astype(np.float32)

@torch.no_grad()
def clip_image_embed(paths, batch_size=16):
    feats = []
    cur_imgs, cur_idx = [], []
    def _flush(imgs):
        if not imgs: return None
        batch = clip_proc(images=imgs, return_tensors="pt")
        batch = {k: v.to(_device) for k, v in batch.items()}
        out = clip_model.get_image_features(**batch)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.detach().cpu().numpy().astype(np.float32)
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = None
        if img is not None:
            cur_imgs.append(img); cur_idx.append(i)
        if len(cur_imgs) == batch_size:
            emb = _flush(cur_imgs); feats.append((cur_idx[:], emb))
            cur_imgs, cur_idx = [], []
    if cur_imgs:
        emb = _flush(cur_imgs); feats.append((cur_idx[:], emb))
    if not feats: return np.zeros((0, 512), dtype=np.float32)
    dim = feats[0][1].shape[1]
    out = np.zeros((len(paths), dim), dtype=np.float32)
    for idxs, emb in feats: out[np.array(idxs)] = emb
    return out

def cos_sim_matrix(A, B):
    if A.size == 0 or B.size == 0: return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return (A @ B.T).astype(np.float32)

# ===================== 8) 메인 파이프라인 =====================
def main():
    print("PDF     :", PDF_PATH.resolve())
    print("OUT_DIR :", OUT_DIR.resolve())
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # 1) PDF + OCR
    doc_meta = parse_pdf(PDF_PATH)
    print("Pages parsed:", doc_meta["page_count"])

    # 2) 이미지 추출
    xref_to_path = export_images(PDF_PATH)
    for p in doc_meta["pages"]:
        for im in p["images"]:
            im["path"] = xref_to_path.get(im["xref"])
    print("Exported images:", len(xref_to_path))

    # 3) 캡션 합성
    path_to_caption = {}
    for p in doc_meta["pages"]:
        for im in p["images"]:
            if not im.get("path"): continue
            cap = extract_caption_for_image(p, im["bbox"])
            placeholder = None
            if not cap and ALWAYS_ADD_PLACEHOLDER_IF_EMPTY:
                placeholder = Path(im["path"]).stem.replace("_", " ")
            abs_img_path = Path(im["path"]).resolve()
            if draw_caption_on_image(abs_img_path, cap, placeholder):
                path_to_caption[abs_img_path.as_posix()] = cap or (placeholder or "")

    # 4) manifest 저장
    for p in doc_meta["pages"]:
        save_json(PAGES_DIR / f"page_{p['page']+1:04d}.json", p)
    manifest = {
        "doc_name": DOC_NAME, "pdf": str(PDF_PATH), "out_dir": str(OUT_DIR),
        "page_count": doc_meta["page_count"], "pages": [], 
        "images_total": sum(len(p["images"]) for p in doc_meta["pages"]),
    }
    for p in doc_meta["pages"]:
        txt_path = PAGES_DIR / f"page_{p['page']+1:04d}.txt"
        try: txt_sz = txt_path.stat().st_size
        except Exception: txt_sz = 0
        needs_ocr = (len(p["words"]) < 5) or (txt_sz < 10)
        manifest["pages"].append({
            "page": p["page"], "png": p["png"],
            "json": str(PAGES_DIR / f"page_{p['page']+1:04d}.json"),
            "txt": str(txt_path), "words": len(p["words"]),
            "images": len(p["images"]), "needs_ocr": bool(needs_ocr),
            "used_ocr": bool(p.get("used_ocr", False)),
        })
    save_json(MANIFEST, manifest)
    print("Manifest saved")

    # 5) 이미지 후보 수집
    def _rel_from_images_dir(p: Path) -> str:
        try: return str(p.resolve().relative_to(OUT_DIR.resolve())).replace("\\", "/")
        except Exception:
            s = str(p).replace("\\", "/")
            prefix = str(OUT_DIR).replace("\\", "/").rstrip("/") + "/"
            return s[len(prefix):] if s.startswith(prefix) else s

    all_img_cands = []
    img_files = sorted([p for p in IMAGES_DIR.glob("*")
                        if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".webp")])
    for f in img_files:
        rel = None
        if SAVE_WITH_SUFFIX:
            cap_f = f.with_name(f.stem + "_cap" + f.suffix)
            rel = _rel_from_images_dir(cap_f if cap_f.exists() else f)
        else:
            rel = _rel_from_images_dir(f)
        try:
            with Image.open(OUT_DIR / rel) as _im:
                w, h = _im.size
            if w <= 220 or h <= 220:
                continue
        except Exception:
            pass
        all_img_cands.append({"path": rel, "bbox": None, "page": None})
    print(f"[info] collected {len(all_img_cands)} images")

    # 6) 임베딩/매칭
    abs_img_paths = [(OUT_DIR / c["path"]).resolve() for c in all_img_cands]
    img_embs = clip_image_embed(abs_img_paths) if abs_img_paths else np.zeros((0,512), np.float32)

    if FUSE_WITH_CAPTION and path_to_caption:
        captions = [(OUT_DIR / c["path"]).resolve().as_posix() for c in all_img_cands]
        cap_texts = [path_to_caption.get(p, "") for p in captions]
        if any(bool(x.strip()) for x in cap_texts):
            txt_feats = clip_text_embed([cap if cap else "" for cap in cap_texts])
            if txt_feats.shape[0] == img_embs.shape[0]:
                a = float(FUSE_IMG_WEIGHT); b = 1.0 - a
                fused = a * img_embs + b * txt_feats
                norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-9
                img_embs = (fused / norms).astype(np.float32)

    pages_compact = []
    for p in doc_meta["pages"]:
        words = [w for w in p.get("words", []) if len((w.get("text") or "")) >= MIN_WORD_LEN]
        q_texts = [w["text"] for w in words]
        dim = img_embs.shape[1] if img_embs.size else 512
        q_clip = clip_text_embed(q_texts) if q_texts else np.zeros((0, dim), dtype=np.float32)

        img_matches_per_word = [[] for _ in range(len(words))]
        if q_clip.size > 0 and img_embs.size > 0:
            sims = (q_clip @ img_embs.T).astype(np.float32)
            for i in range(sims.shape[0]):
                row = sims[i]
                idxs = [j for j, s in enumerate(row) if s >= SIM_THRESHOLD]
                pairs = [(float(row[j]), j) for j in idxs]
                pairs.sort(key=lambda x: x[0], reverse=True)
                top = pairs[:TOP_K]
                img_matches_per_word[i] = [
                    {"score": s, "path": all_img_cands[j]["path"], "bbox": None, "page": None}
                    for (s, j) in top
                ]

        pages_compact.append({
            "page": p["page"],
            "png": str(Path(p["png"]).resolve().relative_to(OUT_DIR.resolve())).replace("\\", "/"),
            "width": p["width"], "height": p["height"],
            "words": [{"text": words[i]["text"], "bbox": words[i]["bbox"], "images": img_matches_per_word[i]}
                      for i in range(len(words))]
        })

    save_json(INDEX_JSON, {"pages": pages_compact})
    print("index.json saved →", INDEX_JSON.resolve())

    # 7) 뷰어 HTML 저장
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>CLIP Overlay Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
body { margin:0; font-family:system-ui, sans-serif; }
.wrap { display:flex; height:100vh; }
.left { width:70%; border-right:1px solid #ddd; padding:12px; overflow:auto; box-sizing:border-box; }
.right { flex:1; padding:12px; overflow:auto; box-sizing:border-box; }
.canvas { position:relative; background-size:100% 100%; background-repeat:no-repeat; background-position:top left; }
.wbtn { position:absolute; background: transparent; border:1px solid transparent; cursor:pointer; outline:none; user-select:none; -webkit-tap-highlight-color: transparent; }
.wbtn:hover { border-color: rgba(0,0,0,0.25); box-shadow:0 0 0 2px rgba(0,0,0,0.06) inset; }
.placeholder { color:#888; }
img { max-width:100%; height:auto; display:block; }
.card { border:1px solid #eee; border-radius:10px; padding:10px; margin-bottom:10px; box-shadow:0 4px 16px rgba(0,0,0,.06); background:#fff; }
.score { color:#444; font-size:12px; margin-bottom:6px; }
.toolbar { display:flex; gap:8px; align-items:center; margin:0 0 10px 0; }
.btn { appearance:none; border:1px solid #ddd; background:#f7f7f7; border-radius:8px; padding:6px 10px; font-size:13px; cursor:pointer; }
.btn:hover { background:#efefef; }
.actions { display:flex; gap:8px; margin-top:8px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
      <label for="pageSel">Page:</label>
      <select id="pageSel"></select>
    </div>
    <div id="stage"></div>
  </div>
  <div class="right">
    <div class="toolbar">
      <h3 style="margin:0;flex:1;">Images only</h3>
      <button id="clearBtn" class="btn" title="오른쪽에 쌓인 이미지 초기화">초기화</button>
    </div>
    <div id="matches"><p class='placeholder'>단어를 클릭하세요.</p></div>
  </div>
</div>
<script>
let DATA=null;
const sel=document.getElementById('pageSel');
const stage=document.getElementById('stage');
const matches=document.getElementById('matches');
const clearBtn=document.getElementById('clearBtn');
function getParam(name, def){const u=new URL(location.href);return u.searchParams.get(name)??def;}
const JSON_URL=getParam('json','index.json');

async function boot(){
  try{
    const res=await fetch(JSON_URL,{cache:'no-store'});
    DATA=await res.json();
    init();
  }catch(e){
    console.error(e);
    stage.innerHTML='<p class="placeholder">index.json 로드 실패</p>';
  }
}
function init(){
  const frag=document.createDocumentFragment();
  for(const p of (DATA.pages||[])){
    const opt=document.createElement('option');
    opt.value=String(p.page);
    opt.textContent='Page '+(p.page+1);
    frag.appendChild(opt);
  }
  sel.appendChild(frag);
  clearBtn.addEventListener('click',()=>matches.innerHTML="<p class='placeholder'>단어를 클릭하세요.</p>");
  sel.addEventListener('change',()=>renderPage(parseInt(sel.value)));
  if(DATA.pages?.length) renderPage(DATA.pages[0].page);
  else stage.innerHTML='<p class="placeholder">No pages</p>';
}
function makeOverlayHTML(p){
  if(!(p.png&&p.width&&p.height)) return '<p class="placeholder">No PNG available.</p>';
  return `<div class="canvas" id="canvas" data-page="${p.page}" style="background-image:url('${p.png}');"></div>`;
}
function placeButtons(p, canvas, pngW, pngH){
  const sx=pngW/p.width, sy=pngH/p.height;
  let html='';
  for(const it of p.words){
    const b=it.bbox;
    const l=Math.max(0,Math.round(b[0]*sx));
    const t=Math.max(0,Math.round(b[1]*sy));
    const wpx=Math.max(1,Math.round((b[2]-b[0])*sx));
    const hpx=Math.max(1,Math.round((b[3]-b[1])*sy));
    const payload=encodeURIComponent(JSON.stringify({word:it.text,images:it.images}));
    html+=`<button class="wbtn" style="left:${l}px;top:${t}px;width:${wpx}px;height:${hpx}px" data-payload="${payload}" title="${it.text}"></button>`;
  }
  const canvasEl=document.getElementById('canvas');
  canvasEl.innerHTML=html;
  canvasEl.querySelectorAll('.wbtn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const payload=JSON.parse(decodeURIComponent(btn.dataset.payload));
      appendMatches(payload);
    });
  });
}
function basename(p){if(!p)return'image.png';const parts=p.split('?')[0].split('#')[0].split('/');return parts[parts.length-1]||'image.png';}
async function triggerDownload(url,filename){
  try{
    const res=await fetch(url);const blob=await res.blob();
    const objectUrl=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=objectUrl;a.download=filename||basename(url);
    document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(objectUrl);
  }catch(e){window.open(url,'_blank');}
}
function appendMatches(payload){
  const imgs=(payload.images||[]);
  if(imgs.length===0){
    if(!matches.querySelector('.card')) matches.innerHTML="<p class='placeholder'>관련 이미지 없음 (threshold 미만)</p>";
    else{
      const emptyCard=document.createElement('div'); emptyCard.className='card';
      emptyCard.innerHTML="<div class='score'>관련 이미지 없음 (threshold 미만)</div>";
      matches.appendChild(emptyCard);
    }
    return;
  }
  const placeholder=matches.querySelector('.placeholder'); if(placeholder) matches.innerHTML='';
  const frag=document.createDocumentFragment();
  imgs.forEach(im=>{
    const fname=basename(im.path);
    const card=document.createElement('div'); card.className='card';
    card.innerHTML=`
      <div class="score">score: ${Number(im.score||0).toFixed(3)}</div>
      <img src="${im.path}" alt="${fname}"/>
      <div class="actions">
        <button class="btn savebtn" data-src="${im.path}" data-fn="${fname}">이미지 저장</button>
      </div>`;
    frag.appendChild(card);
  });
  matches.appendChild(frag);
  matches.querySelectorAll('.savebtn').forEach(b=>{
    if(b.dataset._wired) return; b.dataset._wired='1';
    b.addEventListener('click',()=>{
      const src=b.getAttribute('data-src'); const fn=b.getAttribute('data-fn')||basename(src);
      triggerDownload(src,fn);
    });
  });
}
function renderPage(pno){
  const p=(DATA.pages||[]).find(x=>x.page==pno);
  if(!p){ stage.innerHTML='<p class="placeholder">No page</p>'; return; }
  stage.innerHTML=makeOverlayHTML(p);
  const canvas=document.getElementById('canvas');
  const img=new Image();
  img.onload=function(){
    canvas.style.width=img.naturalWidth+'px';
    canvas.style.height=img.naturalHeight+'px';
    placeButtons(p, canvas, img.naturalWidth, img.naturalHeight);
  };
  img.src=p.png;
}
boot();
</script>
</body>
</html>"""
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("Viewer saved →", HTML_OUT.resolve())


    # 8) 로컬 서버 자동 오픈
    serve_and_open(OUT_DIR, HTML_OUT.name)

# ===================== 9) 로컬 서버 오픈 (WSL: wslview 지원) =====================
def free_port(preferred=8000, max_tries=20):
    p = preferred
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))  # WSL에서 윈도우 localhost로 포워딩
                return p
            except OSError:
                p += 1
    raise RuntimeError("no free port")

def serve_and_open(root: Path, entry_html: str):
    class RootHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            from urllib.parse import unquote, urlsplit
            path = urlsplit(path).path
            path = unquote(path)
            p = (root / path.lstrip("/")).resolve()
            return str(p)

    port = free_port(8000)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), RootHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    url = f"http://localhost:{port}/{entry_html}"  # 윈도우에서 접근용

    print(f"\n[Local Server] {url}")

    # WSL이면 wslview로 윈도우 브라우저 열기
    is_wsl = "microsoft" in platform.release().lower() or "microsoft" in platform.version().lower()
    opened = False
    if is_wsl and shutil.which("wslview"):
        try:
            subprocess.Popen(["wslview", url])
            opened = True
        except Exception:
            pass
    if not opened:
        # 일반적 브라우저 오픈 (WSL에서도 기본 브라우저가 뜰 수 있음)
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    print("Press Ctrl+C to stop server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()

if __name__ == "__main__":
    main()
