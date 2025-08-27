from object_detection import detect_objects_in_image
from pathlib import Path
import cv2

# Test with your war image (change this path!)
image_path = Path("/Users/vullnetvoca/Desktop/tank.jpg")

if image_path.exists():
    print(f"Testing with: {image_path}")
    detections = detect_objects_in_image(image_path)

    print(f"\n🎯 Found {len(detections)} objects!")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class']}: {det['confidence']:.3f} confidence")
        print(f"   Bounding box: {det['bbox']}")
        print(f"   Center: {det['center']}")
else:
    print(f"❌ Image not found: {image_path}")
    print("Please update the path to point to one of your war images!")