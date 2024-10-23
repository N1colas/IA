from PIL import Image
import sys


def estValide(x,y):
    if(x>=0 and y>=0 and x < width and y < height):
        return True
    else:
        return False

def ajouter(x,y,telleDistance):
    if estValide(x,y):
        if carte[x][y]==0:
                a_parcourir.append([x,y,telleDistance])

def rechercher(x,y,telleval):
    result = False;
    result = compare_val(x - 1, y,telleval)
    if (result != False): return result
    result=compare_val(x+1,y,telleval)
    if(result!=False): return result
    result = compare_val(x, y-1,telleval)
    if (result != False): return result
    result=compare_val(x,y+1,telleval)
    if(result!=False): return result


def compare_val(x,y,telleval):
    if (estValide(x, y)):
        if (carte[x][y] == telleval):
            return x,y
        else: return False
    else: return False


im = Image.open('carte.bmp')
pix = im.load()
print(im.size)


width,height = im.size

carte = [[0] * height for _ in range(width)]

for x in range(width):
    for y in range(height):
        r,g,b = (pix[x,y])
        if(r<128 and g<128 and b <128):
            carte[x][y] = -1 #mur
        else:
            carte[x][y] = 0

a_parcourir = [[0,0,0]]

while len(a_parcourir)>0:
    x,y,distance=a_parcourir.pop()
    if(carte[x][y] < distance):
        carte[x][y]=distance
    ajouter(x+1, y,distance+1)
    ajouter(x-1, y,distance+1)
    ajouter(x, y+1,distance+1)
    ajouter(x, y-1,distance+1)

print(carte)

#0 255
#0 distance

x = width-1
y = height-1
val = carte[x][y]
print("val",val)

while(val>1):
    resultat=rechercher(x, y, val-1)
    if (resultat != None):
        x,y=resultat
        pix[x,y]=(255,0,0)
    val=val-1




#for x in range(width):
#    for y in range(height):
#        if(carte[x][y]>0):
#            pix[x,y]=(255,0,0)


im.save('solution.bmp')  # Save the modified pixels as .png
