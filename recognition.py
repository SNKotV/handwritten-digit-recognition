import cv2
import numpy as np
from keras.models import model_from_json
import tkinter as tk
from PIL import Image, ImageDraw

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

window = tk.Tk()
canvas_width = 180
canvas_height = 180
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

filename = 'image.jpg'
image = Image.new('RGB', (18, 18), 'white')
draw = ImageDraw.Draw(image)


def paint(event):
    black = "#000000"
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill=black)
    x1, y1, x2, y2 = x1 // 10, y1 // 10, x2 // 10, y2 // 10
    draw.ellipse([x1, y1, x2, y2], fill=black)
    image.save(filename)


def clear(event):
    canvas.delete('all')
    white = '#FFFFFF'
    draw.rectangle([0, 0, 18, 18], fill=white)
    label.configure(text='Result: ')

def process(event):
    img = cv2.imread('./image.jpg')
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    digit = cv2.resize(thresh, (18, 18))
    digit = np.pad(digit, ((5, 5), (5, 5)), 'constant', constant_values=0)
    prediction = model.predict(digit.reshape(1, 28, 28, 1))
    label.configure(text='Result: ' + str(format(np.argmax(prediction))))


canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', process)
canvas.bind('<ButtonRelease-3>', clear)

label = tk.Label(window, text='Result: ', font=('Helvetica', 32))
label.pack()

window.mainloop()
