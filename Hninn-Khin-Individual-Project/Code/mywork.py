### this wasn't used in our group project in the end ,however,
#  I tried using PIL package to open our image folder to resize them
from PIL import Image
from IPython.display import display

for i in os.listdir(dir):
    name = os.path.basename(i)
    l = os.path.splitext(name)[0]
    image_id.append(i)
    open_image_file = Image.open(file)
    opening_file = open_image_file.resize((width, height), Image.ANTIALIAS)
    a= np.array(opening_file)
    allImage[p - 1] = a
    p= p+1
    x = labels.loc[labels["ImageID"] == l, "Class"]
    Class_order.append(x)

    if opening_file == (width, height, 3):
        number_images_shape = number_images_shape + 1
        a = np.array(opening_file)
        allImage[p - 1] = a
        p = p + 1
        x = labels.loc[labels["ImageID"] == l, "Class"]
        Class_order.append(x)
