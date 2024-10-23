import cv2
#cap = cv2.VideoCapture("http://192.168.1.15:4747/video")
cap = cv2.VideoCapture(0)
while(True):
	# Capture image par image
	ret, frame = cap.read()
	# Operation sur l'image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Affichage du résultat
	cv2.imshow('frame',frame)
	# Touche « q » pour quitter
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
# Fin du programme
cap.release()
cv2.destroyAllWindows()