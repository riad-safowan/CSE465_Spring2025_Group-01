# Facial Expression Recognition with CNN
This project aims to build a lightweight model for real-time facial expression recognition on mobile devices. Using Convolutional Neural Networks (CNN), the system processes facial images and classifies them into emotion categories.

## Contribution Table

| Name | Student ID | Contributions |
|------|------------|---------------|
| Riad Safowan | 2112312642 | • Dataset preparation and preprocessing<br>• Data augmentation implementation<br>• Model architecture design<br>• Training pipeline development |
| Mirza Abir | 2111252642 | • Model optimization for mobile devices<br>• Implementation of real-time inference<br>• Performance benchmarking<br>• Documentation and testing |

Both team members collaborated on experimental design, result analysis, and project report preparation.

## Data Augmentation

The dataset (FER-2013) was expanded by 30% using the following augmentation techniques:

1. **Random Rotation**: ±15° rotation to simulate different face angles
2. **Random Shifting**: Up to 10% translation in any direction
3. **Horizontal Flipping**: Mirror images to balance directional features
4. **Random Erasing**: 5-10 pixel patches removed to simulate occlusion

Each augmented image underwent exactly one of these transformations, randomly selected. This approach enhances model robustness for real-world facial expression recognition on mobile devices.
