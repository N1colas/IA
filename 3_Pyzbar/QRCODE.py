#pip install pyzbar
#https://pypi.org/project/pyzbar/

from pyzbar.pyzbar import decode
from PIL import Image
print(decode(Image.open('qrcode1.png')))