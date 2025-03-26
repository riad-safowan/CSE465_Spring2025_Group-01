# Data Augmentation

The dataset (FER-2013) was expanded by 30% using the following augmentation techniques:

1. **Random Rotation**: ±15° rotation to simulate different face angles
2. **Random Shifting**: Up to 10% translation in any direction
3. **Horizontal Flipping**: Mirror images to balance directional features
4. **Random Erasing**: 5-10 pixel patches removed to simulate occlusion

Each augmented image underwent exactly one of these transformations, randomly selected. This approach enhances model robustness for real-world facial expression recognition on mobile devices.