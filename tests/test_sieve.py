from PIL import Image
from src.sieves import compute_dhash, hamming_distance

img1 = Image.open("data/raw/copydays/original/207600.jpg")
img2 = Image.open("data/raw/copydays/strong/207601.jpg")

h1 = compute_dhash(img1)
h2 = compute_dhash(img2)

print("Hash 1:", h1)
print("Hash 2:", h2)
print("Hamming distance:", hamming_distance(h1, h2))
