## Reliability Characterization for N-version Object Detection
This repository contains the necessary scripts and methodology to reproduce the evaluation results presented in the paper titled, "Reliability Characterization for N-version Object Detection. \
The project introduces two novel object-wise reliability metrics, Cov_OD (Coverage of error in Object Detection) and Cer_OD (Certainty of accurate prediction in Object Detection), designed specifically for N-version object detection systems. The evaluation compares these metrics against conventional metrics (Cov, Cer, mAP) under various model and sensor diversity configurations.

### Prerequisites
1.  Environment\
SetupIt's highly recommended to use a virtual environment. The evaluation script itself primarily relies on CPU power and standard data science libraries (no dedicated GPU is required for the analysis phase).

~~~Bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
~~~

2. Dependencies\
Install the required Python packages
~~~Bash
pip install pandas numpy tqdm
~~~

3. Source Code\
The script relies on the following file structure for the Python classes and modules:.
~~~
├── main.py
├── dataset
│   ├── detectionresult
|   └── labels
├── Integrator/
│   ├── affirmative_integrator.py
│   ├── majority_integrator.py
│   └── unanimous_integrator.py
├── Metrics/
│   ├── Cov.py
│   ├── Cer.py
│   ├── CovOD.py
│   ├── CerOD.py
│   └── mAP.py
└── processor/
    └── dataset.py  # Contains the Dataset class and data loading/analysis logic
~~~

4. Dataset\
The script expects the Ground Truth (GT) and pre-calculated detection results to be organized under the ./dataset directory. The data used in the paper was collected from the CARLA simulator (Town03). The dataset includes 2000 images captured by multiple cameras. 
~~~
./dataset/
├── labels/
│   └── Town03/front/     # Ground Truth (GT)annotations for all frames (used by gt_dir)
└── detectionresult/
    └── labels/
        ├── rtdetr/front/    
        ├── ssd/front/  
        …     
        └── yolov8n/
            ├── front/        # Camera 1 (Center)
            ├── left_1/       # Camera 2
            ├── right_1/      # Camera 3
            ├── left_2/       # Camera 4
            └── right_2/      # Camera 5
~~~
### How to Reproduce Results
1. Execute main.py
~~~Bash
python main.py
~~~

2. Output\
The script executes and saves the results in the ./result directory: