import cv2
import numpy as np
import MNIST_predict

MARGIN = 5


def tag_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    img2, ctrs = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(bw)
    rects = []
    try:
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    except:
        print('None')
    for rect in rects:
        x, y, w, h = rect
        hw = float(h) / w
        if (h > 50) & (w > 50) & (0.6 < hw) & (5 > hw):
            if hw > 1:
                l = h
                x -= int((h-w)/2)
            else:
                l = w
                y += int((w-h) / 2)
            if (x > 20) & (y > 20):
                try:
                    resized_img = cv2.resize(img2[y - MARGIN:y + l + MARGIN, x - MARGIN:x + l + MARGIN], (28, 28))
                except:
                    print(x)
                    print(y - MARGIN, y + l + MARGIN, x - MARGIN, x + l + MARGIN)
                else:
                    data = (np.asarray([resized_img.reshape((28, 28, 1))], 'float32') / 256)
                    predictions = MNIST_predict.predict(data)[0]
                    prediction = np.argmax(predictions)
                    if (np.amax(predictions) > 0.5):
                        cv2.rectangle(img, (x - MARGIN, y - MARGIN), (x + l + MARGIN, y + l + MARGIN), (255, 255, 0),
                                     1)
                        cv2.putText(img, "{}".format(str(prediction)), (x - MARGIN, y - MARGIN - 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255))
    return img


def resize(rawimg):
    fx = 28.0 / rawimg.shape[0]
    fy = 28.0 / rawimg.shape[1]
    fx = fy = min(fx, fy)
    img = cv2.resize(rawimg, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    outimg = np.ones((28, 28), dtype=np.uint8) * 255
    w = img.shape[1]
    h = img.shape[0]
    x = int((28 - w) / 2)
    y = int((28 - h) / 2)
    outimg[y:y+h, x:x+w] = img
    return outimg


def start_real_time():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, img = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', tag_image(img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

start_real_time()