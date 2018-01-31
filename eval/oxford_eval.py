import os
import cv2
import matplotlib.pyplot as plt


def get_query_image_filenames(GT_DIR):
    files = os.listdir(GT_DIR)
    filename_set = set()
    for filename in files:
        if filename[-4:] != ".txt":
            continue
        filename = filename.replace(".txt", "")\
                            .replace("_ok", "") \
                            .replace("_good", "") \
                            .replace("_junk", "") \
                            .replace("_query", "")

        # print(filename)
        filename_set.add(filename)

    query_names = list(filename_set)
    query_names.sort()
    return query_names

def prepare_query_images(query_names, GT_DIR, OUTPUT_QUERY_IMAGE_DIR, OUTPUT_QUERY_IMAGE_LIST_PATH, crop=True, show_image=False):
    with open(OUTPUT_QUERY_IMAGE_LIST_PATH, "w") as list_file:
        for query_name in query_names:
            with open(os.path.join(GT_DIR, query_name+"_query.txt")) as f:
                query_info = f.readline().strip().split(" ")
            # print(query_info)

            # TODO: how to crop with float values? do we need interpolation?
            x_st = int(float(query_info[1]))
            y_st = int(float(query_info[2]))
            x_en = int(float(query_info[3]))
            y_en = int(float(query_info[4]))

            image_name = query_info[0].replace("oxc1_", "")
            image_path = os.path.join(IMAGE_DIR, image_name+".jpg")
            # print("open image:", image_path)
            img_bgr = cv2.imread(image_path)
            
            if crop:
                crop_img_bgr = img_bgr[y_st:y_en, x_st:x_en]                
                img_bgr = crop_img_bgr

            query_filename = query_name+".png"
            query_image_path = os.path.abspath(os.path.join(OUTPUT_QUERY_IMAGE_DIR, query_filename))
            cv2.imwrite(query_image_path, img_bgr)
            list_file.write(query_filename+"\n")
            
            if show_image:                
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                plt.figure()
                plt.imshow(img)
                plt.show()