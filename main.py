import tkinter as tk
from tkinter import filedialog

import numpy as np
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
