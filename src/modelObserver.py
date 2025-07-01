# Aluno: João Lucas Almeida Caldas Ferraz
# Número USP: 12609651
# Disciplina: Processamento de Imagens (SCC0251) - 2025/1
# Projeto Final - Filtragem de Ruído em Imagens de Mamografia Digital

import numpy as np
import os
import math
import csv
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from scipy.fft import fft2, ifft2
from scipy.linalg import pinv
from sklearn.metrics import roc_auc_score

# Geração dos polinômios de Laguerre até a ordem J
def laguerre_polynomial(x, J):
    L = np.zeros((len(x), J + 1))
    for j in range(J + 1):
        for k in range(j + 1):
            c = (math.factorial(j) /
                 (math.factorial(k) * math.factorial(j - k))) * ((-x) ** k) / math.factorial(k)
            L[:, j] += c
    return L

# Geração de canais Laguerre-Gaussianos 2D baseados no raio r
def laguerre_gaussian_2d(r, J, h):
    L = laguerre_polynomial(2 * np.pi * r ** 2 / h ** 2, J)
    u = L * np.exp(-np.pi * r ** 2 / h ** 2)[:, np.newaxis]
    scale = np.sqrt(2) / h
    u *= scale
    return u

# Geração do conjunto completo de canais Laguerre-Gaussianos 2D para a imagem
def generate_channels(nx, ny, nch, ch_width):
    xi = np.arange(nx) - (nx - 1) / 2
    yi = np.arange(ny) - (ny - 1) / 2
    xxi, yyi = np.meshgrid(xi, yi, indexing='ij')
    r = np.sqrt(xxi ** 2 + yyi ** 2).flatten()
    u = laguerre_gaussian_2d(r, nch - 1, ch_width)
    u = u.reshape(nx, ny, nch)
    return u

# Carrega todas as imagens DICOM de uma pasta em um array 3D
def load_dicom_folder(path, max_files=None):
    files = sorted([f for f in os.listdir(path) if f.endswith('.dcm')])
    if max_files is not None:
        files = files[:max_files]

    images = []
    for f in files:
        ds = pydicom.dcmread(os.path.join(path, f), force=True)
        if 'TransferSyntaxUID' not in ds.file_meta:
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        arr = ds.pixel_array.astype(np.float64)
        images.append(arr)
    return np.stack(images, axis=-1)

# Implementação do Model Observer CHO com canais de Laguerre-Gauss
def conv_LG_CHO_2d(trimg_sa, trimg_sp, testimg_sa, testimg_sp, ch_width, nch, b_conv=1):
    nx, ny, ntr_sa = trimg_sa.shape
    _, _, ntr_sp = trimg_sp.shape
    _, _, nte_sa = testimg_sa.shape
    _, _, nte_sp = testimg_sp.shape

    u = generate_channels(nx, ny, nch, ch_width)
    sig_mean = np.mean(trimg_sp, axis=2) - np.mean(trimg_sa, axis=2)

    # Convolução opcional dos canais com a média da diferença de sinal
    if b_conv:
        ch_sig = np.zeros_like(u)
        for ich in range(nch):
            U_fft = fft2(u[:, :, ich])
            S_fft = fft2(sig_mean)
            conv = ifft2(np.abs(U_fft) ** 2 * S_fft) / (nx * ny)
            conv = np.real(conv)
            conv /= np.sqrt(np.sum(conv ** 2))
            ch_sig[:, :, ich] = conv
    else:
        ch_sig = u

    ch = ch_sig.reshape(nx * ny, nch)

    # Projeção das imagens de treino nos canais
    tr_sa_ch = np.zeros((nch, ntr_sa))
    for i in range(ntr_sa):
        img_vec = trimg_sa[:, :, i].reshape(1, nx * ny)
        tr_sa_ch[:, i] = (img_vec @ ch).flatten()

    tr_sp_ch = np.zeros((nch, ntr_sp))
    for i in range(ntr_sp):
        img_vec = trimg_sp[:, :, i].reshape(1, nx * ny)
        tr_sp_ch[:, i] = (img_vec @ ch).flatten()

    # Cálculo do template do observador
    s_ch = np.mean(tr_sp_ch, axis=1) - np.mean(tr_sa_ch, axis=1)
    k_sa = np.cov(tr_sa_ch)
    k_sp = np.cov(tr_sp_ch)
    k = (k_sa + k_sp) / 2

    w = s_ch @ pinv(k)

    # Projeção das imagens de teste
    te_sa_ch = np.zeros((nch, nte_sa))
    for i in range(nte_sa):
        img_vec = testimg_sa[:, :, i].reshape(1, nx * ny)
        te_sa_ch[:, i] = (img_vec @ ch).flatten()

    te_sp_ch = np.zeros((nch, nte_sp))
    for i in range(nte_sp):
        img_vec = testimg_sp[:, :, i].reshape(1, nx * ny)
        te_sp_ch[:, i] = (img_vec @ ch).flatten()

    # Geração de scores com adição de ruído para simular variabilidade humana
    t_sa = w @ te_sa_ch
    t_sp = w @ te_sp_ch

    std_ta = np.std(t_sa)
    std_tp = np.std(t_sp)

    t_sa = t_sa + np.random.normal(0, std_ta * 6.6, size=t_sa.shape)
    t_sp = t_sp + np.random.normal(0, std_tp * 6.6, size=t_sp.shape)

    # Cálculo do SNR e AUC
    snr = (np.mean(t_sp) - np.mean(t_sa)) / np.sqrt((np.var(t_sp) + np.var(t_sa)) / 2)

    labels = np.concatenate([np.zeros_like(t_sa), np.ones_like(t_sp)])
    scores = np.concatenate([t_sa, t_sp])
    auc = roc_auc_score(labels, scores)

    tplimg = (w @ ch.T).reshape(nx, ny)
    chimg = ch_sig
    meanSP = np.mean(trimg_sp, axis=2)
    meanSA = np.mean(trimg_sa, axis=2)
    meanSig = sig_mean
    k_ch = k

    return snr, auc, t_sp, t_sa, chimg, tplimg, meanSP, meanSA, meanSig, k_ch

# Executa o experimento para um grupo de imagens (fundo e sinal) simulando n leitores
def run_experiment(signal_folder, background_folder, ch_width=1.5, nch=8, ntrain=300, nreaders=30):
    saroi_fd = load_dicom_folder(background_folder)
    sproi_fd = load_dicom_folder(signal_folder)

    saroi_fd = saroi_fd[:, :, 1:]
    sproi_fd = sproi_fd[:, :, 1:]

    nx, ny, nsa = saroi_fd.shape
    _, _, nsp = sproi_fd.shape

    snrs = []
    aucs = []

    max_train_samples = min(2000, ntrain) 
    ntrain = max_train_samples

    for r in range(nreaders):
        np.random.seed(r)
        cases_sa_temp = np.arange(5, nsa, 6)
        np.random.shuffle(cases_sa_temp)
        cases_sa = []
        for c in cases_sa_temp:
            if c - 5 >= 0:
                cases_sa.extend([c, c - 1, c - 2, c - 3, c - 4, c - 5])
        cases_sa = np.array(cases_sa)

        cases_sp = cases_sa_temp // 6

        # Ajuste para garantir que treino e teste existam e não ultrapassem o limite
        if len(cases_sa) < 6 * 2:
            print(f"[Reader {r+1}] Não há dados suficientes para treino e teste, pulando.")
            continue

        id_sa_tr = cases_sa[:ntrain * 6]
        id_sp_tr = cases_sp[:ntrain]
        id_sa_test = cases_sa[ntrain * 6:]
        id_sp_test = cases_sp[ntrain:]

        id_sa_tr = id_sa_tr[id_sa_tr < nsa]
        id_sp_tr = id_sp_tr[id_sp_tr < nsp]
        id_sa_test = id_sa_test[id_sa_test < nsa]
        id_sp_test = id_sp_test[id_sp_test < nsp]

        # Checar se conjunto teste tem amostras
        if len(id_sa_test) == 0 or len(id_sp_test) == 0:
            print(f"[Reader {r+1}] Teste vazio, pulando.")
            continue

        trimg_sa = saroi_fd[:, :, id_sa_tr]
        trimg_sp = sproi_fd[:, :, id_sp_tr]
        testimg_sa = saroi_fd[:, :, id_sa_test]
        testimg_sp = sproi_fd[:, :, id_sp_test]

        try:
            snr, auc, *_ = conv_LG_CHO_2d(trimg_sa, trimg_sp, testimg_sa, testimg_sp, ch_width, nch)
            snrs.append(snr)
            aucs.append(auc)
            print(f"Reader {r+1}/{nreaders} - SNR: {snr:.3f} AUC: {auc:.3f}")
        except Exception as e:
            print(f"[Reader {r+1}] Erro: {e}")

    snrs = np.array(snrs)
    aucs = np.array(aucs)

    if len(snrs) > 0:
        print(f"\nMean SNR: {snrs.mean():.3f} ± {snrs.std():.3f}")
        print(f"Mean AUC: {aucs.mean():.3f} ± {aucs.std():.3f}")
    else:
        print("\nNenhum resultado válido obtido.")

    return snrs, aucs

# Caminhos base
base_calcs = "C:\\Users\\Johnny\\Documents\\SCC0251\\images\\roi\\calcs"
base_no_calcs = "C:\\Users\\Johnny\\Documents\\SCC0251\\images\\roi\\no_calcs"

# Lista de grupos avaliados
grupos = [
    "or/0.5-0.5",
    "or/1-1",
    "or/2-2",
    "mb/0.5-1/gaussian",
    "mb/0.5-1/bilateral",
    "mb/0.5-1/tv",
    "mb/1-2/gaussian",
    "mb/1-2/bilateral",
    "mb/1-2/tv"
]

# Lista para armazenar os resultados
resultados = []

# Avaliação
for grupo in grupos:
    signal_path = os.path.join(base_calcs, grupo)
    background_path = os.path.join(base_no_calcs, grupo)

    if not os.path.exists(signal_path):
        print(f"[SKIP] Sinal não encontrado: {signal_path}")
        continue
    if not os.path.exists(background_path):
        print(f"[SKIP] Fundo não encontrado: {background_path}")
        continue

    print(f"\n>>> Avaliando grupo: {grupo}")
    snrs, aucs = run_experiment(signal_path, background_path, ch_width=1.5, nch=8, ntrain=200, nreaders=30)

    if len(aucs) > 0:
        resultados.append({
            "Grupo": grupo,
            "AUC_média": np.mean(aucs),
            "AUC_std": np.std(aucs)
        })
    else:
        resultados.append({
            "Grupo": grupo,
            "AUC_média": None,
            "AUC_std": None
        })

# Salva os resultados em CSV
csv_path = "C:\\Users\\Johnny\\Documents\\SCC0251\\results\\CHO_results.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

with open(csv_path, mode='w', newline='') as csvfile:
    fieldnames = ["Grupo", "AUC_média", "AUC_std"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in resultados:
        writer.writerow(row)

print(f"\nResumo salvo em: {csv_path}")