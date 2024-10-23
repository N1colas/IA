#VOIR : https://github.com/lincolnloop/python-qrcode
#

import qrcode
img = qrcode.make('Cette formation est fantastique')
#type(img)  

img.save("monqrcode.png") #enregistre l'image

img.show() #affiche l'image dans une fenêtre


##AVANCE

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('https://www.wikipedia.fr')
qr.make(fit=True)

img = qr.make_image(fill_color="green", back_color="cyan")
img.show();