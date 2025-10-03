import argparse
import cv2
import pandas as pd

# Argparse
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

# Read Image
img = cv2.imread(img_path)

# Load CSV
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# Globals
clicked = False
r = g = b = xpos = ypos = 0

# Mouse Callback
def draw_function(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

# Function to get color name
def getColorName(R, G, B):
    minimum = 10000
    cname = "Unknown"
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# Loop
while True:
    cv2.imshow("image", img)
    if clicked:
        # Draw rectangle
        cv2.rectangle(img, (20,20), (750,60), (b,g,r), -1)

        # Text
        text = getColorName(r,g,b) + ' R='+ str(r) + ' G='+ str(g) + ' B='+ str(b)

        # Put text
        cv2.putText(img, text, (50,50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # For light colors -> black text
        if (r+g+b >= 600):
            cv2.putText(img, text, (50,50), 2, 0.8, (0,0,0), 2, cv2.LINE_AA)

        clicked = False

    # Exit with ESC
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
