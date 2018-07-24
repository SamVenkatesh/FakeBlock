import cv2
import Constants
from BuildTrainTestCNN import NNModel

face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')

'''
We need to isolate all faces in the image and retrieve
the one with the largest "area". 
Crop/transform it to network specs and return it.
'''


def format_image(image_to_format):
    if len(image_to_format.shape) > 2 and image_to_format.shape[2] == 3:
        image_to_format = cv2.cvtColor(image_to_format, cv2.COLOR_BGR2GRAY)
    else:
        image_to_format = cv2.imdecode(image_to_format, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    detected_faces = face_cascade.detectMultiScale(
        image_to_format,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize = (48, 48),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # If we don't find a face, return None
    if not len(detected_faces) > 0:
        return None
    max_face = detected_faces[0]
    for face in detected_faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face

    # Chop image to face
    face = max_face
    image_to_format = image_to_format[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to fit network specs
    try:
        image_to_format = cv2.resize(image_to_format, (Constants.FACE_SIZE, Constants.FACE_SIZE),
                                     interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("Image resize exception. Check input resolution inconsistency.")
        return None
    return image_to_format


video_capture = cv2.VideoCapture(0)
nnModel = NNModel()
nnModel.build_model()
nnModel.model.load_weights('model_weights')


'''
Pulled this infinite loop of pulling the WebCam feed straight from OpenCVs docs.
Its a little choppy on my computer, your mileage may vary.
'''


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    result = nnModel.make_prediction(format_image(frame))
    print(result)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if result is not None:
        for index, emotion in enumerate(Constants.EMOTIONS):
            cv2.putText(frame, emotion, (15, index * 20 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (130, index * 20 + 10), (130 +
                                                          int(result[0][index] * 100), (index + 1) * 20 + 4),
                          (255, 0, 0), -1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
