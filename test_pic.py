import pickle
import cv2

model_path = 'model.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

labels_dict = {0: 'trang', 1: 'vang'}

image_path = 'mm.jpg'

frame = cv2.imread(image_path)

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Flatten image to 1D vector
img_flat = img_rgb.flatten()

# Make prediction
predicted_label = model.predict([img_flat])
predicted_character = labels_dict[int(predicted_label[0])]

cv2.putText(frame, f'Predicted label: {predicted_label[0]}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(frame, f'Predicted character: {predicted_character}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Image', frame)
cv2.waitKey(0)

cv2.destroyAllWindows()
