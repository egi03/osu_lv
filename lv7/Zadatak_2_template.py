import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
def run_task1_4(img_path):
    img = Image.imread(img_path)

    # prikazi originalnu sliku
        
    """     plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()  """

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img =  img.astype(np.float64) / 255 if img_path != "imgs\\test_4.jpg" else img.astype(np.float64)

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape

    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()  
    print(img_array_aprox)

    unique_colors = np.unique(img_array, axis=0)
    print("\nBroj jedinstvenih boja u slici:", len(unique_colors))


    def run_kmeans(n, X):
        k_means = KMeans(n_clusters=n, random_state=0)
        k_means.fit(X)
        y_kmeans = k_means.predict(X)
        plt.figure()
        plt.scatter(X[::500,0], X[::500,1], c=y_kmeans[::500], cmap='viridis')
        plt.title("Broj klastera: " + str(n))

    def create_image(n, img_array):
        k_means = KMeans(n_clusters=n, random_state=0)
        k_means.fit(img_array)
        
        centroids = k_means.cluster_centers_
        
        labels = k_means.predict(img_array)
        img_array_aprox = centroids[labels]
        img_aprox = img_array_aprox.reshape((w, h, d))

        print("centroi", centroids)
        print("labels:", labels)
        
        
        fig, ax = plt.subplots(2)
        ax[1].imshow(img_aprox)
        ax[0].imshow(img)
        ax[0].set_title("Originalna slika")
        ax[1].set_title(f"Slika sa {n} boja")
        plt.tight_layout()
        plt.show()

    create_image(3, img_array)
    create_image(5, img_array)
    plt.show()

def run_task6():
    img = Image.imread("imgs\\test_1.jpg")
        
    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img =  img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape

    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()  

    unique_colors = np.unique(img_array, axis=0)
    print("\nBroj jedinstvenih boja u slici:", len(unique_colors))


    k_dots = [2,3,4,5,6,7]
    j_dots = []
    for k in [2,3,4,5,6,7]:
        k_means = KMeans(n_clusters=k, random_state=0)
        k_means.fit(img_array)
        y_kmeans = k_means.predict(img_array)
        j_dots.append(k_means.inertia_)

    plt.figure()
    plt.plot(k_dots, j_dots)
    plt.show()


def run_task7():
    img = Image.imread("imgs\\test_1.jpg")
        
    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img =  img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape

    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()  

    unique_colors = np.unique(img_array, axis=0)
    print("\nBroj jedinstvenih boja u slici:", len(unique_colors))



    k_means = KMeans(n_clusters=3, random_state=0)
    k_means.fit(img_array)
    centroids = k_means.cluster_centers_
    labels = k_means.predict(img_array)
    img_array_aprox = centroids[labels]




    filtered_imgs = []
    for k in range(3):
        new_img = img_array_aprox.copy()
        for i, _ in enumerate(img_array_aprox):
            if labels[i] == k:
                new_img[i] = np.array([0,0,0])
            else:
                new_img[i] = np.array([1,1,1])

        img_aprox = new_img.reshape((w, h, d))

        print("centroi", centroids)
        print("labels:", labels)
        
        fig, ax = plt.subplots(2)
        ax[1].imshow(img_aprox)
        ax[0].imshow(img)
        ax[0].set_title("Originalna slika")
        ax[1].set_title(f"{k}. boja")
        plt.tight_layout()
        plt.show()
        filtered_imgs.append(img_aprox)


    fig, ax = plt.subplots(4)
    ax[0].imshow(img)
    ax[1].imshow(filtered_imgs[0])
    ax[2].imshow(filtered_imgs[1])
    ax[3].imshow(filtered_imgs[2])
    plt.tight_layout()
    plt.show()


run_task1_4("imgs\\test_1.jpg")
run_task1_4("imgs\\test_2.jpg")
run_task1_4("imgs\\test_3.jpg")
run_task1_4("imgs\\test_4.jpg")
run_task1_4("imgs\\test_5.jpg")
run_task1_4("imgs\\test_6.jpg")
run_task6()
run_task7()