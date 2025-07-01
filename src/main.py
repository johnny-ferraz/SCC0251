# Aluno: João Lucas Almeida Caldas Ferraz
# Número USP: 12609651
# Disciplina: Processamento de Imagens (SCC0251) - 2025/1
# Projeto Final - Filtragem de Ruído em Imagens de Mamografia Digital

from filters import filters
from saveROIs import saveROIs
from scipy.io import loadmat

# Função para extrair e salvar ROIs de imagens sem agrupamento de microcalcificações
def getAbsentROIs(noCalcsPath, ROInoCalcsPath, absCoordinates, numROIs, doses, Lambda, Sigma, Tau, ph):
    # Carrega a imagem ground-truth completa
    GT_mat = loadmat(noCalcsPath + f'gt\\{ph}.mat')
    GT = GT_mat['GT']
    
    # Para cada condição de dose e modo, carrega a imagem ruidosa completa correspondente
    for low, high, mode in doses:
        mat = loadmat(f'{noCalcsPath}\\or\\{low}\\{ph}.mat')
        imgNoisy = mat['imgNoisy']
        
        # Se o modo for 'mb', aplica o filtro gaussiano, bilateral e total variation
        if mode == 'mb':
            imgGaussian, imgBilateral, imgTV = filters(imgNoisy, Lambda, Sigma, Tau)
        
            # Para cada ROI, salva as versões filtradas nas pastas correspondentes
            for roi in range(1, numROIs + 1):
                saveROIs(imgGaussian, f'{ROInoCalcsPath}{mode}\\{low}-{high}\\gaussian\\{(ph-1)*numROIs + roi}.dcm', absCoordinates[roi-1])
                saveROIs(imgBilateral, f'{ROInoCalcsPath}{mode}\\{low}-{high}\\bilateral\\{(ph-1)*numROIs + roi}.dcm', absCoordinates[roi-1])
                saveROIs(imgTV, f'{ROInoCalcsPath}{mode}\\{low}-{high}\\tv\\{(ph-1)*numROIs + roi}.dcm', absCoordinates[roi-1])
        else:
            # Para outros modos, salva as ROIs da imagem ruidosa e da ground-truth
            for roi in range(1, numROIs + 1):
                saveROIs(imgNoisy, f'{ROInoCalcsPath}{mode}\\{low}-{high}\\{(ph-1)*numROIs + roi}.dcm', absCoordinates[roi-1])
                saveROIs(GT, f'{ROInoCalcsPath}gt\\{(ph-1)*numROIs + roi}.dcm', absCoordinates[roi-1])

# Função para extrair e salvar ROIs de imagens com agrupamento de microcalcificações
def getPresentROIs(calcsPath, ROICalcsPath, presentCoordinates, numROIs, doses, Lambda, Sigma, Tau, ph):
    
    # Mapa com a quantidade de ROIs para cada imagem no phantom
    rois_per_image_map = {
        1: [8]*25,
        2: [10]*20,
        3: [10]*20,
        4: [7]*19 + [4],
        5: [6]*19 + [2],
        6: [10]*20,
        7: [10]*20,
        8: [8]*24 + [7] + [1],
        9: [7]*19 + [4],
        10: [10]*20
    }

    rois_per_image = rois_per_image_map[ph]
    roi_index = 0

    for i, rois_in_image in enumerate(rois_per_image, 1): 
        # Carrega a imagem ground-truth e ruidosa específica
        GT_mat = loadmat(f'{calcsPath}\\gt\\{i}.mat')
        GT = GT_mat['GT']

        # Para cada condição de dose e modo, carrega a imagem ruidosa completa correspondente
        for low, high, mode in doses:
            mat = loadmat(f'{calcsPath}\\or\\{low}\\{i}.mat')
            imgNoisy = mat['imgNoisy']

            # Se o modo for 'mb', aplica o filtro gaussiano, bilateral e total variation
            if mode == 'mb':
                imgGaussian, imgBilateral, imgTV = filters(imgNoisy, Lambda, Sigma, Tau)

                # Para cada ROI, salva as versões filtradas nas pastas correspondentes
                for r in range(rois_in_image):
                    idx = roi_index + r + 1  # Índice global da ROI
                    coord = presentCoordinates[idx - 1]

                    saveROIs(imgGaussian, f'{ROICalcsPath}{mode}\\{low}-{high}\\gaussian\\{(ph-1)*numROIs + idx}.dcm', coord)
                    saveROIs(imgBilateral, f'{ROICalcsPath}{mode}\\{low}-{high}\\bilateral\\{(ph-1)*numROIs + idx}.dcm', coord)
                    saveROIs(imgTV, f'{ROICalcsPath}{mode}\\{low}-{high}\\tv\\{(ph-1)*numROIs + idx}.dcm', coord)

            else:
                # Para outros modos, salva as ROIs da imagem ruidosa e da ground-truth
                for r in range(rois_in_image):
                    idx = roi_index + r + 1
                    coord = presentCoordinates[idx - 1]

                    saveROIs(imgNoisy, f'{ROICalcsPath}{mode}\\{low}-{high}\\{(ph-1)*numROIs + idx}.dcm', coord)
                    saveROIs(GT, f'{ROICalcsPath}gt\\{(ph-1)*numROIs + idx}.dcm', coord)

        # Atualiza índice global das ROIs
        roi_index += rois_in_image
                
def main():
    # Número de phantoms, ROIs e doses simuladas
    # numPhantoms = 10
    numPhantoms = 1
    numROIs = 200
    doses = [[0.5, 0.5, 'or'], [1, 1, 'or'], [2, 2, 'or'], [0.5, 1, 'mb'], [1, 2, 'mb']]

    # Parâmetros do equipamento
    parameters = loadmat('C:\\Users\\Johnny\\Documents\\SCC0251\\parameters\\Parameters_Hologic_FFDM.mat')
    Lambda = parameters['lambda']
    Sigma = parameters['sigma']
    Tau = parameters['tau']

    for ph in range(1, numPhantoms + 1):
        print(f'\nPhantom {ph}')
        
        # Paths das imagens completas
        noCalcsPath = f'C:\\Users\\Johnny\\Documents\\SCC0251\\test_images\\complete\\Phantom_{ph}\\no_calcs\\'
        calcsPath = f'C:\\Users\\Johnny\\Documents\\SCC0251\\test_images\\complete\\Phantom_{ph}\\calcs\\'
        
        # Paths para salvar ROIs
        ROInoCalcsPath = f'C:\\Users\\Johnny\\Documents\\SCC0251\\test_images\\roi\\no_calcs\\'
        ROICalcsPath = f'C:\\Users\\Johnny\\Documents\\SCC0251\\test_images\\roi\\calcs\\'
        
        # Carrega as coordenadas das ROIs sem microcalcificações
        absentPositions_mat = loadmat(noCalcsPath + 'absent_positions.mat')
        absentPositions = absentPositions_mat['abs_SimulationInfo']
        absCoordinates = [absentPositions[0, i].squeeze().tolist() for i in range(absentPositions.shape[1])]
        
        # Carrega as coordenadas das ROIs com microcalcificações
        presentPositions_mat = loadmat(calcsPath + 'present_positions.mat')
        presentPositions = presentPositions_mat['present_SimulationInfo']
        presentCoordinates = [presentPositions[0, i].squeeze().tolist() for i in range(presentPositions.shape[1])]

        # Extrai e salva ROIs
        getAbsentROIs(noCalcsPath, ROInoCalcsPath, absCoordinates, numROIs, doses, Lambda, Sigma, Tau, ph)
        getPresentROIs(calcsPath, ROICalcsPath, presentCoordinates, numROIs, doses, Lambda, Sigma, Tau, ph)

if __name__ == "__main__":
    main()