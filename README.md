# ScoutAI

ScoutAI is a computer vision-based analytics system for tracking football drills. It detects the ball, cones, player movement, speed, body lean, and ball touches using YOLO and MediaPipe.

## Setup Instructions
### Step 1: Create a Virtual Environment

Create a virtual environment (you can choose your own name for it):

python -m venv venv_<your_custom_name>


Activate the virtual environment:

Windows:
```
.\venv_<your_custom_name>\Scripts\Activate
```
Linux / MacOS:
```
source venv_<your_custom_name>/bin/activate
```
### Step 2: Install Dependencies

Install required packages:
```
pip install -r requirements.txt
```


Install PyTorch (CPU version) if not included in requirements.txt:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Run the Project
```
python main.py
```

#### To change the input video, place your video file in the Data folder.

#### To use a different pre-trained model for cone detection, place it in the models folder and update the model_path variable in main.py.
