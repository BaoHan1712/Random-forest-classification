import pickle
import cv2
import time

model_path = 'model.pickle'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

labels_dict = {0: 'trang', 1: 'vang'}

pTime = time.time()
frames_count = 0

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.flatten()

    # Make prediction
    predicted_label = model.predict([img_flat])
    predicted_character = labels_dict[int(predicted_label[0])]

    cv2.putText(frame, f'Predicted label: {predicted_label[0]}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Predicted character: {predicted_character}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cTime = time.time()
    frames_count += 1
    if cTime - pTime >= 1:
        fps = frames_count / (cTime - pTime)
        frames_count = 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        # Hiển thị FPS trên cửa sổ terminal
        print(f'FPS: {int(fps)}')

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
