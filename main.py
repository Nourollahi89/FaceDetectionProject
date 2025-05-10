import cv2
import matplotlib.pyplot as plt

# خواندن تصویر
image_bgr = cv2.imread("saman.jpg")

# تبدیل رنگ به RGB
image_rgb = cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2RGB)

# نمایش تصویر اصلی
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# بارگذاری مدل تشخیص چهره
model = cv2.CascadeClassifier("model.xml")

# تشخیص چهره‌ها
faces = model.detectMultiScale(image_bgr)
print("Detected faces:", faces)

# رسم کادر سبز دور چهره‌ها
if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 4)

    plt.imshow(image_rgb)
    plt.title("Detected Face (Green Box)")
    plt.axis('off')
    plt.show()
else:
    print("هیچ چهره‌ای شناسایی نشد.")
