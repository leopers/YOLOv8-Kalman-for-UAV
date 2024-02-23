# YOLOv8 + Kalman

Esse projeto consiste no treinamento e utilização de uma rede neural do tipo YOLOv8 para detecção de marcos fiduciais para propósitos UAV.
O Tracking dos referidos marcos é realizado através da detecção destes e previsão cinemática utilizando um Filtro de Kalman.
O dataset de treino foi capturado por uma câmera de celular e passou por um processo de data augmentation.

## Estrutura do repositório 
```
.
└── EXAME-CM203/  
    ├── /prototype               (some useless testing)
    ├── /src/
    │   ├── /training/
    │   │   └── yolo_training.py (Notebook usado para treino do modelo)
    │   ├── YOLOdetector.py      (Detector de marcos fiduciais)
    │   ├── Kalman.py            (Filtro de Kalman; implementação)
    │   └── wKF.py               (Detecção + Previsão + Atualização)
    └── /utils/
        ├── /Training_Dataset    (Dataset de treino, validação e teste)
        ├── best.pt              (Modelo treinado com o dataset acima)
        └── marco_pouso.png      (Modelo de marco fiducial)
```
## Para rodar o projeto
```
pip install ultralytics
pip install numpy
pip install opencv-python
```

Para executar e ver o tracking funcionando basta executar o arquivo **src/wKF.py**
