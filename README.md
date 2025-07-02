# Projeto Final – Filtragem de Ruído em Imagens de Mamografia Digital  

---

## Descrição

Este projeto tem como objetivo avaliar o desempenho de diferentes técnicas de filtragem de ruído em imagens simuladas de **mamografia digital**, com foco na detecção de **agrupamentos de microcalcificações** sob diferentes condições de dose. O *pipeline* completo inclui a simulação das imagens, aplicação de filtros, extração de regiões de interesse (ROIs) e avaliação com um modelo de observador computacional.

---

## Geração do Dataset
As imagens utilizadas foram geradas sinteticamente por um algoritmo desenvolvido no **Laboratório de Visão Computacional (LAVI/USP)**, implementado em MATLAB. Como entrada, o algoritmo recebe anatomias simuladas no *software* de ensaios virtuais VICTRE [1] e projetadas com um método de *ray-tracing* baseado no algoritmo de *Siddon* [5]. Com base nessas imagens, o algoritmo do LAVI realiza:

- Inserção de **agrupamentos de microcalcificações** em regiões com pelo menos 80% de tecido denso [3,4];
- Simulação de ruído **Poisson-Gaussiano** em diferentes níveis de dose, utilizando o **modelo de variância afim** descrito em [2].

---

## Organização e Extração de ROIs

- Para imagens sem lesão, basta gerar uma única imagem e extrair diversas ROIs em diferentes regiões;
- Para imagens com lesão, o agrupamento de microcalcificações deve ser inserido antes da simulação do ruído, pois a densidade anatômica muda de imagem para imagem;
- Por esse motivo, foram necessárias múltiplas imagens simuladas com diferentes posicionamentos do *cluster* para totalizar as 200 ROIs com lesão utilizadas neste trabalho.

---

## Técnicas de Filtragem Aplicadas

Antes da aplicação dos filtros, as imagens passam por uma **Transformada de Estabilização de Variância (VST)** baseada na **Transformada Generalizada de Anscombe**, permitindo tratar o ruído Poisson-Gaussiano no domínio dos filtros gaussianos.

Os seguintes filtros foram aplicados no domínio estabilizado:

- **Filtro Gaussiano** (`scipy.ndimage.gaussian_filter`)  
- **Filtro Bilateral** (`skimage.restoration.denoise_bilateral`)  
- **Filtro Total Variation (TV)** (`skimage.restoration.denoise_tv_chambolle`)  

Após a filtragem, aplica-se a **inversão fechada e não enviesada da VST** para retornar ao domínio original da imagem [6].

---

## Avaliação com *Model Observer*

A avaliação da eficácia dos filtros foi realizada utilizando o **Channelized Hotelling Observer (CHO)**, um modelo computacional utilizado para simular o desempenho de observadores humanos em tarefas de detecção.

A implementação considera:

- **Canais Laguerre-Gaussianos 2D**, que atuam como filtros seletivos de frequência;
- Cálculo de métricas como **SNR (Signal-to-Noise Ratio)** e **AUC (Área sob a Curva ROC)**;
- Separação em conjuntos de **treinamento** e **teste**, com simulação de diversos "leitores";
- Adição de ruído aos escores para simular a **variabilidade intraobservador**.

Esse modelo permite quantificar a capacidade de identificar a presença de lesões sob diferentes condições de ruído e filtragem, sendo amplamente utilizado em estudos de percepção em imagens médicas.

---

## 📚 Bibliografia

[1] **BADANO, A. et al.** *Evaluation of digital breast tomosynthesis as replacement of full-field digital mammography using an in silico imaging trial.* JAMA Network Open, v. 1, n. 7, p. e185474, 2018.  
[2] **BORGES, L. R. et al.** *Noise models for virtual clinical trials of digital breast tomosynthesis.* Med Phys., v. 46, n. 6, p. 2683–9, 2019.  
[3] **BORGES, L. R.; MARQUES, P. M. A.; VIEIRA, M. A. C.** *A 2-AFC study to validate artificially inserted microcalcification clusters in digital mammography.* In: NISHIKAWA, R. M.; SAMUELSON, F. W. (ed.). *Medical Imaging 2019: Image Perception, Observer Performance, and Technology Assessment.* Proc. SPIE, v. 10952, p. 178–184, 2019.  
[4] **BORGES, L. R. et al.** *Effect of denoising on the localization of microcalcification clusters in digital mammography.* In: BOSMANS, H.; MARSHALL, N.; ONGEVAL, C. V. (ed.). *15th International Workshop on Breast Imaging.* Proc. SPIE, v. 11513, p. 115130K, 2020.  
[5] **MACHADO, A. Y.; MASSERA, R. T.; TOMAL, A.** *Pipeline to generate synthesized mammographic images: reliability of a new framework for data augmentation-based ray-tracing method, monte carlo simulation, and deep learning scatter estimation.* In: *Medical Imaging 2025: Physics of Medical Imaging.* SPIE, v. 13405, p. 821–826, 2025.  
[6] **MÄKITALO, M.; FOI, A.** *Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise.* IEEE Transactions on Image Processing, v. 22, n. 1, p. 91–103, 2012.  
