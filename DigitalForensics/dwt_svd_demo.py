import argparse, os
from pathlib import Path
import numpy as np
import cv2
import pywt
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim

# =============== Util: tipovi, konverzije, DWT/SVD ==================
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_float01(img):
    return img.astype(np.float32) / 255.0

def to_uint8(img01):
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)

def rgb2ycc(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

def ycc2rgb(img_ycc):
    return cv2.cvtColor(img_ycc, cv2.COLOR_YCrCb2RGB)

def dwt2(channel, wavelet='haar'):
    return pywt.dwt2(channel, wavelet)

def idwt2(coeffs, wavelet='haar'):
    return pywt.idwt2(coeffs, wavelet)

def svd(mat):
    return np.linalg.svd(mat, full_matrices=False)  # U, S, Vt

# =============== Metrike =======================
def psnr_y(img1_rgb, img2_rgb):
    y1 = to_float01(rgb2ycc(img1_rgb)[:, :, 0])
    y2 = to_float01(rgb2ycc(img2_rgb)[:, :, 0])
    return _psnr(y1, y2, data_range=1.0)

def ssim_y(img1_rgb, img2_rgb):
    y1 = to_float01(rgb2ycc(img1_rgb)[:, :, 0])
    y2 = to_float01(rgb2ycc(img2_rgb)[:, :, 0])
    return _ssim(y1, y2, data_range=1.0)

def nc(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a = (a - a.mean())/(a.std()+1e-8); b = (b - b.mean())/(b.std()+1e-8)
    return float((a*b).mean())

# =============== Napadi =======================
def attack_jpeg(img_rgb, quality=50):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY),
    int(quality)])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def attack_gaussian_noise(img_rgb, sigma=5):
    n = np.random.normal(0, sigma, img_rgb.shape).astype(np.float32)
    noisy = np.clip(img_rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return noisy

def attack_blur(img_rgb, k=5):
    return cv2.GaussianBlur(img_rgb, (int(k)|1, int(k)|1), 0)

def attack_resize(img_rgb, scale=0.5):
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back

def attack_rotate_small(img_rgb, angle=2):
    h, w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rot

# =============== DWT+SVD Embed/Extract (Y kanal) =======================
def embed_watermark(cover_rgb, wm_img, alpha=0.04, wavelet='haar', subband='LL', adapt_alpha=True):
    # 1) Y kanal
    ycc = rgb2ycc(cover_rgb)
    Y = to_float01(ycc[:, :, 0])

    # 2) DWT
    LL, (LH, HL, HH) = dwt2(Y, wavelet)
    bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
    B = bands[subband]

    # 3) Priprema žiga (u gray, skaliran na dimenzije podopsega)
    if wm_img.ndim == 3:
        wm_gray = cv2.cvtColor(wm_img, cv2.COLOR_RGB2GRAY)
    else:
        wm_gray = wm_img
    wm_gray = cv2.resize(wm_gray, (B.shape[1], B.shape[0]), interpolation=cv2.INTER_AREA)
    W = to_float01(wm_gray)

    # 4) SVD žiga i podopsega
    UW, SW, VWt = svd(W)
    UH, SH, VHt = svd(B)

    # 5) (Opcioni) adaptivni alpha na odnos medijana singularnih vrednosti
    if adapt_alpha:
        medH = float(np.median(SH))
        medW = float(np.median(SW)) if np.median(SW) > 0 else float(SW.mean() + 1e-8)
        alpha_eff = alpha * (medH / (medW + 1e-8))
    else:
        alpha_eff = alpha

    # 6) Ugradnja preko S_H
    S_emb = SH + alpha_eff * SW[:len(SH)]
    B_emb = (UH * S_emb) @ VHt

    # 7) Zamena podopsega i IDWT
    if subband == 'LL':
        LL2, LH2, HL2, HH2 = B_emb, LH, HL, HH
    elif subband == 'LH':
        LL2, LH2, HL2, HH2 = LL, B_emb, HL, HH
    elif subband == 'HL':
        LL2, LH2, HL2, HH2 = LL, LH, B_emb, HH
    else:  # 'HH'
        LL2, LH2, HL2, HH2 = LL, LH, HL, B_emb

    Y_emb = idwt2((LL2, (LH2, HL2, HH2)), wavelet)
    Y_emb = np.clip(Y_emb, 0.0, 1.0)

    out = ycc.copy()
    out[:, :, 0] = to_uint8(Y_emb)
    wm_rgb = ycc2rgb(out)

    meta = {
        'alpha': float(alpha),
        'alpha_eff': float(alpha_eff),
        'wavelet': wavelet,
        'subband': subband,
        'SH': SH.tolist(),
    }
    # Vraćamo i U_W, S_W, V_W^T radi kasnije rekonstrukcije
    return wm_rgb, meta, (UW, SW, VWt)

def extract_watermark(wm_rgb, cover_rgb, meta, UW, SW, VWt):
    alpha_eff = float(meta.get('alpha_eff', meta['alpha']))
    wavelet = meta['wavelet']; subband = meta['subband']
    SH = np.array(meta['SH'], dtype=np.float32)

    Yw = to_float01(rgb2ycc(wm_rgb)[:, :, 0])
    LLw, (LHw, HLw, HHw) = dwt2(Yw, wavelet)

    bands_w = {'LL': LLw, 'LH': LHw, 'HL': HLw, 'HH': HHw}
    Bw = bands_w[subband]

    # SVD napadnutog podopsega
    Uw, Sw, Vtw = svd(Bw)

    # Procena S žiga
    SW_hat = (Sw[:len(SH)] - SH) / (alpha_eff + 1e-12)

    # Rekonstrukcija W^ pomoću U_W i V_W^T
    r = min(UW.shape[0], VWt.shape[0], SW_hat.shape[0])
    UW_r = UW[:, :r]
    VWt_r = VWt[:r, :]
    SW_r = np.diag(SW_hat[:r])
    W_hat = UW_r @ SW_r @ VWt_r
    W_hat = np.clip(W_hat, 0.0, 1.0)
    return (W_hat * 255).astype(np.uint8)


# =============== MAIN (jedan tok embed→attack→extract) =======================
def main():
    parser = argparse.ArgumentParser(description='DWT+SVD watermark demo(single-file)')
    parser.add_argument('--cover', default='data/cover.png', help='putanja do cover slike (RGB)')
    parser.add_argument('--wm', default='data/watermark.png', help='putanja do watermark slike (gray/kolor)')
    parser.add_argument('--alpha', type=float, default=0.04)
    parser.add_argument('--wavelet', default='haar', choices=['haar','db2'])
    parser.add_argument('--subband', default='LL', choices=['LL','LH','HL','HH'])

    # napadi
    parser.add_argument('--jpeg', type=int, default=50)
    parser.add_argument('--noise', type=float, default=5.0)
    parser.add_argument('--blur', type=int, default=5)
    parser.add_argument('--resize', type=float, default=0.5)
    parser.add_argument('--rot', type=float, default=2.0)
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Učitavanje
    cover_bgr = cv2.imread(args.cover)
    if cover_bgr is None:
        raise FileNotFoundError(args.cover)
    cover_rgb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2RGB)
    wm_img = cv2.imread(args.wm, cv2.IMREAD_UNCHANGED)
    if wm_img is None:
        raise FileNotFoundError(args.wm)
    if wm_img.ndim == 2:
        wm_rgb = cv2.cvtColor(wm_img, cv2.COLOR_GRAY2RGB)
    else:
        wm_rgb = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)

    # Embed
  #  wm_cover_rgb, meta, (UW, SW, VWt) = embed_watermark(
   #     cover_rgb, wm_rgb, alpha=args.alpha, wavelet=args.wavelet,
   #     subband=args.subband, adapt_alpha=True
   # )

    wm_cover_rgb, meta, (UW, SW, VWt) = embed_watermark(
        cover_rgb, wm_rgb,
        alpha=args.alpha, wavelet=args.wavelet, subband=args.subband,
        adapt_alpha=False
    )

    # Transparentnost
    P = psnr_y(cover_rgb, wm_cover_rgb)
    S = ssim_y(cover_rgb, wm_cover_rgb)

    # Sacuvaj watermarked
    out_wm = os.path.join(args.outdir, 'watermarked.png')
    cv2.imwrite(out_wm, cv2.cvtColor(wm_cover_rgb, cv2.COLOR_RGB2BGR))

    # Napadi i ekstrakcija
    results = []

    # 0) Bez napada
    W_hat = extract_watermark(wm_cover_rgb, cover_rgb, meta, UW, SW, VWt)
    W_ref = cv2.resize(cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2GRAY),
                       (W_hat.shape[1], W_hat.shape[0]))
    NC0 = nc(W_ref, W_hat)
    cv2.imwrite(os.path.join(args.outdir, 'extracted_clean.png'), W_hat)
    results.append(('none', P, S, NC0))

    # 1) JPEG
    attacked = attack_jpeg(wm_cover_rgb, quality=args.jpeg)
    W_hat = extract_watermark(attacked, cover_rgb, meta, UW, SW, VWt)
    cv2.imwrite(os.path.join(args.outdir, f'attacked_jpeg{args.jpeg}.jpg'), cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, f'extracted_jpeg{args.jpeg}.png'), W_hat)
    results.append((f'jpeg{args.jpeg}', P, S, nc(W_ref, W_hat)))

    # 2) Gaussian noise
    attacked = attack_gaussian_noise(wm_cover_rgb, sigma=args.noise)
    W_hat = extract_watermark(attacked, cover_rgb, meta, UW, SW, VWt)
    cv2.imwrite(os.path.join(args.outdir, f'attacked_noise{int(args.noise)}.png'), cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, f'extracted_noise{int(args.noise)}.png'), W_hat)
    results.append((f'noise{int(args.noise)}', P, S, nc(W_ref, W_hat)))

    # 3) Blur
    attacked = attack_blur(wm_cover_rgb, k=args.blur)
    W_hat = extract_watermark(attacked, cover_rgb, meta, UW, SW, VWt)
    cv2.imwrite(os.path.join(args.outdir, f'attacked_blur{int(args.blur)}.png'), cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, f'extracted_blur{int(args.blur)}.png'), W_hat)
    results.append((f'blur{int(args.blur)}', P, S, nc(W_ref, W_hat)))

    # 4) Resize
    attacked = attack_resize(wm_cover_rgb, scale=args.resize)
    W_hat = extract_watermark(attacked, cover_rgb, meta, UW, SW, VWt)
    cv2.imwrite(os.path.join(args.outdir, f'attacked_resize{args.resize}.png'), cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, f'extracted_resize{args.resize}.png'), W_hat)
    results.append((f'resize{args.resize}', P, S, nc(W_ref, W_hat)))

    # 5) Rotacija
    attacked = attack_rotate_small(wm_cover_rgb, angle=args.rot)
    W_hat = extract_watermark(attacked, cover_rgb, meta, UW, SW, VWt)
    cv2.imwrite(os.path.join(args.outdir, f'attacked_rot{int(args.rot)}.png'), cv2.cvtColor(attacked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, f'extracted_rot{int(args.rot)}.png'), W_hat)
    results.append((f'rot{int(args.rot)}', P, S, nc(W_ref, W_hat)))

    # Upis metrika
    with open(os.path.join(args.outdir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Transparentnost: PSNR={P:.2f} dB, SSIM={S:.4f}")
        for name, p, s, ncv in results:
            f.write(f"{name:>10s} | NC={ncv:.4f}")

    print(f"Sačuvano u: {args.outdir}")
    print(f"PSNR={P:.2f} dB, SSIM={S:.4f}")
    for name, _, _, ncv in results:
        print(f"{name:>10s} NC={ncv:.4f}")

if __name__ == '__main__':
    np.random.seed(2025)
    main()

