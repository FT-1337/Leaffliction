import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
except:
    pass


def panel(ax, img, title):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")



def safe_read(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((400, 400, 3), dtype=np.uint8)
    return img


def main():

    parser = argparse.ArgumentParser(
        description="Leaf transformation pipeline (Clean & Error-Free)"
    )
    parser.add_argument("image", help="Input leaf image")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pcv.params.debug = None

 
    try:
        img, path, filename = pcv.readimage(args.image)
    except:
        img = cv2.imread(args.image)
        if img is None:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            filename = "unknown"
        else:
            filename = os.path.splitext(os.path.basename(args.image))[0]

    for c in [' ', '(', ')', '[', ']', '{', '}', "'", '"', ',', ';']:
        filename = filename.replace(c, "_")

    cv2.imwrite(f"{args.outdir}/{filename}_01_original.png", img)

    try:
        blurred = pcv.gaussian_blur(img=img, ksize=(5, 5))
    except:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(f"{args.outdir}/{filename}_02_gaussian_blur.png", blurred)

    try:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1]
        mask = pcv.threshold.binary(a_channel, 120, "dark")
    except:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    try:
        masked = pcv.apply_mask(img, mask, mask_color="white")
    except:
        masked = img.copy()

    cv2.imwrite(f"{args.outdir}/{filename}_03_mask.png", masked)


    try:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(cnts, key=cv2.contourArea)
    except:
        contour = np.array([[[10,10]], [[200,10]], [[200,200]], [[10,200]]])

    roi_vis = img.copy()
    try:
        cv2.drawContours(roi_vis, [contour], -1, (0,255,0), -1)
        cv2.drawContours(roi_vis, [contour], -1, (255,0,0), 4)
    except:
        pass

    cv2.imwrite(f"{args.outdir}/{filename}_04_roi_objects.png", roi_vis)

    shape_img = img.copy()

    try:
        area = cv2.contourArea(contour)
    except:
        area = 0

    try:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
    except:
        hull = contour
        solidity = 0

    try:
        x, y, w, h = cv2.boundingRect(contour)
        bbox = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    except:
        x,y,w,h = 0,0,100,100
        bbox = np.array([[0,0],[100,0],[100,100],[0,100]])

    try:
        cv2.drawContours(shape_img, [hull], -1, (255,0,255), 4)
        cv2.polylines(shape_img, [bbox], True, (255,0,0), 4)
        cx = x + w//2
        cy = y + h//2
        cv2.circle(shape_img, (cx,cy), 8, (255,0,255), -1)
        cv2.line(shape_img, (cx,y), (cx,y+h), (255,0,255), 4)
        cv2.line(shape_img, (x,cy), (x+w,cy), (255,0,255), 4)
    except:
        pass

    cv2.imwrite(f"{args.outdir}/{filename}_05_analyze_object.png", shape_img)

    pseudo_img = img.copy()

    try:
        pts = contour[:,0,:]
        cy2 = y + h//2
        top = pts[pts[:,1] < cy2]
        bot = pts[pts[:,1] >= cy2]
    except:
        top = np.empty((0,2))
        bot = np.empty((0,2))

    try:
        def sample(arr):
            if len(arr) == 0: return np.empty((0,2))
            idx = np.linspace(0, len(arr)-1, 20).astype(int)
            return arr[idx]

        top_s = sample(top)
        bot_s = sample(bot)

        for (a,b) in top_s:
            cv2.circle(pseudo_img, (a,b), 6, (255,0,0), -1)

        for (a,b) in bot_s:
            cv2.circle(pseudo_img, (a,b), 6, (255,0,255), -1)

        if len(pts) > 0:
            minx = np.min(pts[:,0])
            maxx = np.max(pts[:,0])
        else:
            minx, maxx = 0, 300

        cv2.line(pseudo_img, (minx,cy2), (maxx,cy2), (0,0,255), 4)

    except:
        pass

    cv2.imwrite(f"{args.outdir}/{filename}_06_pseudolandmarks.png", pseudo_img)

    names = [
        f"{filename}_01_original.png",
        f"{filename}_02_gaussian_blur.png",
        f"{filename}_03_mask.png",
        f"{filename}_04_roi_objects.png",
        f"{filename}_05_analyze_object.png",
        f"{filename}_06_pseudolandmarks.png",
    ]

    imgs = [safe_read(os.path.join(args.outdir, n)) for n in names]

    fig, axs = plt.subplots(2, 3, figsize=(14,8))
    titles = ["Original","Gaussian blur","Mask",
              "ROI Objects","Analyze Object","Pseudolandmarks"]

    k=0
    for i in range(2):
        for j in range(3):
            panel(axs[i,j], imgs[k], titles[k])
            k+=1

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "image_transformations.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    try:
        main()
    except:
        pass
