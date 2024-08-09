# YOLOv8 + Kalman

This project involves training and using a YOLOv8 neural network for detecting fiducial markers for UAV purposes. The tracking of these markers is performed through detection and kinematic prediction using a Kalman Filter. The training dataset was captured using a cellphone camera and underwent a data augmentation process.

## Repo Structure
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
## Running the project
```
pip install ultralytics
pip install numpy
pip install opencv-python
```

To execute and see the tracking in action, just run the src/wKF.py file.
