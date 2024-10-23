#python version Python 3.8.6rc1 (64bits) windows
#pip install Pillow
#note : Pil est inclus dans Pillow

#Nous utilisons ici le package PIL pour manipuler notre image
from PIL import Image

im = Image.open('disque.bmp','r')
#on recupere la largeur et hauteur de notre image
width, height = im.size

#on s'assure du format RGB
im_rgb = im.convert("RGB")

#test de deux pixels de notre image
print("Couleur verte: ",im_rgb.getpixel((25,5))) #retourne 0,255,0
print("Couleur noire: ",im_rgb.getpixel((0,0))) #retourne 0,0,0

width, height = im.size

max_compteur=0
max_y=0
dernier_x=0
max_dernier_x=0

for y in range(im_rgb.height-1):
	compteur=0
	for x in range(im_rgb.width-1):
		r,g,b=im_rgb.getpixel((x,y))

		if(g>128):
			compteur=compteur+1
			dernier_x = x

	if(max_compteur<compteur):
		max_compteur=compteur
		max_y= y
		max_dernier_x = dernier_x

print("Rayon ",max_compteur)
#affiche 9
print("Coordonnee du centre du cercle : ",max_dernier_x-int(max_compteur/2),max_y)
#affiche 25 5


