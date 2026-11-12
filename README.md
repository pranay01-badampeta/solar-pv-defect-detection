# Solar PV Panel Defect Detection and Classification

A web-based system that detects and classifies defects in solar photovoltaic panels from images with a CNN, and detects electrical faults from sensor readings with Random Forest and SVM.

This project accompanies the paper **[Web-based Interface for Detecting and Classifying the Solar PV Panel Defects using CNN Model](https://ieeexplore.ieee.org/document/11399178)**, published at the 2025 IEEE 9th International Conference on Information and Communication Technology (CICT).

## What it does

| Part | Input | Model | Output |
| --- | --- | --- | --- |
| Visual inspection | Panel photo | CNN (custom or ResNet-18 transfer learning) | Clean, Dusty, Bird-drop, Snow-Covered, Electrical-damage, Physical-Damage |
| Electrical monitoring | Irradiance, temperatures, DC voltage and current | Random Forest and SVM | Normal, Partial shading, Soiling, Open circuit, Degradation |
| Web interface | Image upload or sensor form | Flask app | Prediction, confidence and a recommended action |

## Project structure

```
model.py           CNN architectures and image transforms
train_cnn.py       trains and evaluates the image classifier
sensor_faults.py   trains Random Forest and SVM on sensor readings
app.py             Flask web app (templates/index.html)
models/            saved models and metrics (created by the scripts)
```

## Run it

```bash
pip install -r requirements.txt

# 1. Image data: download the Kaggle "Solar Panel Images: Clean and Faulty" dataset
#    into data/images/ (one folder per class), then train the CNN
python train_cnn.py --data data/images --arch cnn --epochs 30
#    or transfer learning, usually more accurate on small datasets
python train_cnn.py --data data/images --arch resnet18 --epochs 15

# 2. Sensor model (simulated readings, or your own labelled CSV)
python sensor_faults.py --simulate

# 3. Web app
python app.py        # open http://127.0.0.1:5000
```

## Results

Paste the numbers printed by the scripts here (also saved in `models/metrics.json` and `models/sensor_metrics.json`):

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| CNN (image) | | |
| Random Forest (sensor) | | |
| SVM (sensor) | | |

The confusion matrix is saved to `models/confusion_matrix.png`.

## Notes

- The sensor data in `--simulate` mode comes from a simple PV module model with injected faults, because labelled field data is rarely public. The same script trains on real readings from a CSV with the columns `irradiance, module_temp, ambient_temp, voltage, current, power, fault`.
- Features are normalised by irradiance (current and power per kW/m², module temperature rise) so the models detect under-performance rather than cloudy weather.

## Tech

Python · PyTorch · torchvision · scikit-learn · Flask · pandas · Matplotlib
