"""
After running this test_script.py script. you will be prompt like,

Enter the path to the image file: 

here are some sample file paths you can use - 
sample/happy.jpg
sample/aug_happy.jpg
sample/surprise.jpg
sample/aug_surprise.jpg
sample/angry.jpg
sample/aug_angry.jpg
sample/disgust.jpg
sample/aug_disgust.jpg
sample/fear.jpg
sample/aug_fear.jpg
sample/neutral.jpg
sample/aug_neutral.jpg
sample/sad.jpg
sample/aug_sad.jpg
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# ResBlock definition (needed to recreate the model architecture)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


# AttentionEmotionNet model definition
class AttentionEmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(AttentionEmotionNet, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Apply attention
        attn = self.attention(x)
        x = x * attn

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def predict_emotion(image_path, model_path):
    """
    Predicts the emotion in an image using the trained AttentionEmotionNet model.

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the saved model weights (.pt file)

    Returns:
        str: Predicted emotion
        float: Confidence score
    """
    # Define emotion classes
    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Create model and load weights
    model = AttentionEmotionNet(num_classes=len(emotion_classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform and prepare image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        pred_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[pred_class].item()

    predicted_emotion = emotion_classes[pred_class]

    # Visualize the prediction
    plt.figure(figsize=(10, 5))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_emotion}\nConfidence: {confidence:.2f}")
    plt.axis('off')

    # Show probabilities
    plt.subplot(1, 2, 2)
    probs = probabilities.cpu().numpy()
    plt.barh(emotion_classes, probs)
    plt.xlim(0, 1)
    plt.title('Emotion Probabilities')
    plt.tight_layout()
    plt.show()

    return predicted_emotion, confidence


if __name__ == "__main__":
    # Change these paths to your model and test image
    MODEL_PATH = 'model.pt'

    # Example usage with a single image
    image_path = input("Enter the path to the image file: ")

    emotion, confidence = predict_emotion(image_path, MODEL_PATH)
    print(f"Predicted emotion: {emotion}")
    print(f"Confidence: {confidence:.4f}")