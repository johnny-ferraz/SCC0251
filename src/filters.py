# Aluno: João Lucas Almeida Caldas Ferraz
# Número USP: 12609651
# Disciplina: Processamento de Imagens (SCC0251) - 2025/1
# Projeto Final - Filtragem de Ruído em Imagens de Mamografia Digital

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle

# Função que aplica a inversão fechada da Transformada de Anscombe Generalizada para reverter o domínio da VST
def gen_anscombe_inverse_closed_form(D, sigma, alpha=1.0, g=0.0):
    sigma_scaled = sigma / alpha

    D_safe = np.maximum(D, 1e-9) # Evita divisões por zero

    # Fórmula de inversão aproximada (Mäkitalo & Foi, IEEE TIP 2012)
    inverse = (D_safe / 2)**2 \
            + (1/4) * np.sqrt(3/2) * (1 / D_safe) \
            - (11/8) * (1 / D_safe**2) \
            + (5/8) * np.sqrt(3/2) * (1 / D_safe**3) \
            - 1/8 - sigma_scaled**2

    inverse = np.maximum(0, inverse) * alpha + g 

    return inverse

# Função para aplicar filtragem da imagem usando VST e IVST
def filters(imgNoisy, Lambda, Sigma, Tau):
    imgNoisy = imgNoisy.astype(np.float32)

    # Aplicação da Variance-Stabilizing Transform (VST)
    s = (imgNoisy - Tau) / Lambda
    fz = 2 * np.sqrt(s + 3/8 + Sigma**2)

    # Aplica os três filtros no domínio estabilizado
    fz_gauss = gaussian_filter(fz, sigma=1) # Filtro Gaussiano
    fz_bilateral = denoise_bilateral(fz, sigma_color=0.05, sigma_spatial=3, channel_axis=None) # Filtro Bilateral
    fz_tv = denoise_tv_chambolle(fz, weight=0.1) # Filtro Total Variation

    # Aplica a IVST para retornar ao domínio original
    img_gauss = gen_anscombe_inverse_closed_form(fz_gauss, sigma=Sigma, alpha=Lambda, g=Tau)
    img_bilateral = gen_anscombe_inverse_closed_form(fz_bilateral, sigma=Sigma, alpha=Lambda, g=Tau)
    img_tv = gen_anscombe_inverse_closed_form(fz_tv, sigma=Sigma, alpha=Lambda, g=Tau)

    return img_gauss, img_bilateral, img_tv