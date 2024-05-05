import tkinter as tk
from tkinter import filedialog

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageTk, ImageOps
from matplotlib import pyplot as plt

global yol
global img_label2
def ana_sayfa():
    print("Ana sayfa")


def odev_1():
    global img_label2

    odev1_pencere = tk.Toplevel()
    odev1_pencere.title("Eşik Değer Değiştirme")

    img_label = tk.Label(odev1_pencere)
    img_label.grid(row=4, column=3, padx=10, pady=10)

    img_label2 = tk.Label(odev1_pencere)
    img_label2.grid(row=4, column=5, padx=10, pady=10)

    select_image = tk.Button(odev1_pencere, text="Resim Seç", command=lambda: selectImage(img_label))
    select_image.grid(row=3, column=3, padx=10, pady=10)

    histogram = tk.Button(odev1_pencere, text="Histogram Göster", command=histogram_goster)
    histogram.grid(row=5, column=3, padx=10, pady=10)

    threshold_slider = tk.Scale(odev1_pencere, from_=0, to=255, orient=tk.HORIZONTAL,
                                label="Eşik Değer", command=update_threshold)
    threshold_slider.grid(row=6, column=3, padx=10, pady=10)
    #odev1_pencere.pack(expand=True, fill="both", padx=50, pady=50)


def update_threshold(value):
    threshold_value = int(value)
    display_image(value)


def display_image(value):
    global yol, img_label2
    org_img = Image.open(yol)
    thresholded_image = apply_threshold(org_img, value)
    thresholded_image = thresholded_image.resize((500, 400), Image.BOX)
    img_tk = ImageTk.PhotoImage(thresholded_image)

    img_label2.config(image=img_tk)
    img_label2.image = img_tk


def apply_threshold(image, threshold):
    gray_image = ImageOps.grayscale(image)
    image_array = np.array(gray_image, dtype=np.uint8)
    thresholded_array = (image_array > np.uint8(threshold)).astype(np.uint8) * 255
    thresholded_image = Image.fromarray(thresholded_array.astype(np.uint8))

    return thresholded_image

def selectImage(label):
    global yol
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500, 400), Image.BOX)
        img = ImageTk.PhotoImage(img)

        yol = file_path

        label.config(image=img)
        label.image = img


def histogram_goster():
    global yol
    resim = Image.open(yol)

    resim_gri = ImageOps.grayscale(resim)

    histogram = resim_gri.histogram()

    plt.plot(histogram, color='gray')
    plt.title('Görüntü Histogramı')
    plt.xlabel('Piksel Değerleri')
    plt.ylabel('Frekans')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_sigmoid_to_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sigmoid_image = sigmoid(image)
    sigmoid_image = (255 * (sigmoid_image - sigmoid_image.min()) / (sigmoid_image.max() - sigmoid_image.min())).astype(
        np.uint8)
    return sigmoid_image


def shifted_sigmoid(x, shift):
    return 1 / (1 + np.exp(-(x - shift)))


def apply_shifted_sigmoid_to_image(image_path, shift):
    # Resmi yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Yatay kaydırılmış sigmoid fonksiyonunu uygula
    shifted_sigmoid_image = shifted_sigmoid(image, shift)

    # Sonuçları 0 ile 255 arasına normalize et
    shifted_sigmoid_image = (255 * (shifted_sigmoid_image - shifted_sigmoid_image.min()) /
                             (shifted_sigmoid_image.max() - shifted_sigmoid_image.min())).astype(np.uint8)

    return shifted_sigmoid_image


def soru1_1():
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    # Kullanıcı bir dosya seçti mi kontrol et
    if input_image_path:
        output_image = apply_sigmoid_to_image(input_image_path)
        cv2.imwrite("sigmoid_image.png", output_image)
        cv2.imshow("Standart Sigmoid Çıktısı", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Resim seçilmedi.")


def soru1_2():
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    shift = float(input("Lütfen kaydırma değerini girin: "))

    # Kullanıcı bir dosya seçti mi kontrol et
    if input_image_path:
        output_image = apply_shifted_sigmoid_to_image(input_image_path, shift)
        cv2.imwrite("sigmoid_image.png", output_image)
        cv2.imshow("Standart Sigmoid Çıktısı", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Resim seçilmedi.")


def sloped_sigmoid(x, slope):
    return 1 / (1 + np.exp(-slope * x))


def apply_sloped_sigmoid_to_image(image_path, slope):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sloped_sigmoid_image = sloped_sigmoid(image, slope)
    sloped_sigmoid_image = (255 * (sloped_sigmoid_image - sloped_sigmoid_image.min()) /
                            (sloped_sigmoid_image.max() - sloped_sigmoid_image.min())).astype(np.uint8)
    return sloped_sigmoid_image


def soru1_3():
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    slope = float(input("Lütfen eğim değerini girin: "))

    if input_image_path:
        output_image = apply_sloped_sigmoid_to_image(input_image_path, slope)
        cv2.imwrite("sigmoid_image.png", output_image)
        cv2.imshow("Standart Sigmoid Çıktısı", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Resim seçilmedi.")


def soru2_1():
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    if input_image_path:
        image = cv2.imread(input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 15, 100, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('Hough Lines', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def soru2_2():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Eye Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_properties(contour, image, mask):
    moments = cv2.moments(contour)
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    length = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    _, (width, height), _ = cv2.minAreaRect(contour)
    diagonal = np.sqrt(width ** 2 + height ** 2)
    energy = cv2.contourArea(contour) / (width * height)
    _, _, angle = cv2.fitEllipse(contour)
    entropy = cv2.contourArea(contour) / (width * height)
    mean_val = cv2.mean(image, mask=mask)
    median_val = cv2.medianBlur(image, 5)
    return center, length, width, height, diagonal, energy, entropy, mean_val, median_val


def soru4_1():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    image = cv2.imread(image_path)

    lower_green = np.array([0, 50, 0])
    upper_green = np.array([100, 255, 100])

    mask = cv2.inRange(image, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    columns = ['No', 'Center', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean', 'Median']
    data = []

    f = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            length = cv2.arcLength(contour, True)
            rect = cv2.minAreaRect(contour)
            diagonal = np.sqrt(rect[1][0] ** 2 + rect[1][1] ** 2)
            energy = -1
            entropy = -1
            try:
                energy = np.sum(np.square(contour))
                entropy = -np.sum((contour / np.sum(contour)) * np.log(np.nan_to_num(contour / np.sum(contour))))
            except:
                print("sıfıra bölme hatası")
            mean = np.mean(contour)
            median = np.median(contour)
            data.append([f, (cx, cy), length, rect[1][0], diagonal, energy, entropy, mean, median])
            f = f + 1

    df = pd.DataFrame(data, columns=columns)

    df.to_excel('yasil_alanlar.xlsx', index=False)
    print("Koyu yeşil alanların özellikleri 'yesil_alanlar.xlsx' dosyasına kaydedildi.")

def main():
    root = tk.Tk()
    root.title("Üst Menü Örneği")

    menubar = tk.Menu(root)
    root.config(menu=menubar)

    ana_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Ana Sayfa", menu=ana_menu)
    ana_menu.add_command(label="Ana Sayfa", command=ana_sayfa)

    odev1_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Ödev 1", menu=odev1_menu)
    odev1_menu.add_command(label="Eşik Değer Değiştirme", command=odev_1)

    soru1_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Soru 1", menu=soru1_menu)
    soru1_menu.add_command(label="Standar Sigmoid Fonksiyonu", command=soru1_1)
    soru1_menu.add_command(label="Yatay Kaydırılmış Sigmoid Fonksiyonu", command=soru1_2)
    soru1_menu.add_command(label="Eğimli Sigmoid Fonksiyonu", command=soru1_3)

    soru2_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Soru 2", menu=soru2_menu)
    soru2_menu.add_command(label="Yol Çizgi Takip", command=soru2_1)
    soru2_menu.add_command(label="Göz Tespit Uygulaması", command=soru2_2)

    soru4_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Soru 4", menu=soru4_menu)
    soru4_menu.add_command(label="Tarla Ürün Tespit Uygulaması", command=soru4_1)

    label1 = tk.Label(root, text="Ders Adı: Dijital Görüntü İşlme")
    label2 = tk.Label(root, text="İsim : Fatih Emirhan Türker")
    label3 = tk.Label(root, text="Numara : 211229055")

    label1.grid(row=0, column=0, padx=10, pady=10)
    label2.grid(row=1, column=0, padx=10, pady=10)
    label3.grid(row=2, column=0, padx=10, pady=10)

    current_font = label1.cget("font")
    new_font = (current_font[0], 16)
    label1.config(font=new_font)
    label2.config(font=new_font)
    label3.config(font=new_font)

    root.mainloop()


if __name__ == "__main__":
    main()
