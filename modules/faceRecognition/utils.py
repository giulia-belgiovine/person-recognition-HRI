from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms, datasets


def format_face_coord(bottle_face_coord):
    import yarp
    """
    Process the face coordinates read by the yarp bottle
    :param bottle_face_coord: coordinates of detected face (yarp bottle)
    :return: list of boxes with box defined as  [top, left, bottom, right]
    """
    list_face_coord = []
    list_face_id = []

    for i in range(bottle_face_coord.size()):
        face_data = bottle_face_coord.get(i).asList()

        list_face_id.append(face_data.get(0).asString())
        face_coordinates = face_data.get(1).asList()

        list_face_coord.append([face_coordinates.get(0).asDouble(), face_coordinates.get(1).asDouble(),
                      face_coordinates.get(2).asDouble(), face_coordinates.get(3).asDouble()])

    return list_face_id, list_face_coord


def get_center_face(faces, width):
    """
    From a list of faces identify the most centered one according to the image width
    :param faces:
    :param width:
    :return: most centered face coordinate
    """
    min_x = 1e5
    min_index = 0

    for i, face_coord in enumerate(faces):
        center_face_x = face_coord[0] + ((face_coord[2] - face_coord[0]) / 2)
        if abs((width / 2)-center_face_x) < min_x:
            min_index = i
            min_x = abs((width / 2)-center_face_x)

    return [faces[min_index]]


def face_alignement(face_coord, frame, margin=40):
    """
    Preprocess tha face with standard face alignment preprocessing
    :param coord_faces: list of faces' coordinates
    :param frame: rgb image
    :return: list face aligned image
    """

    xmin = int(face_coord[0])
    ymin = int(face_coord[1])
    xmax = int(face_coord[2])
    ymax = int(face_coord[3])

    heigth_face = (ymax - ymin)
    width_face = (xmax - xmin)

    center_face = [ymin + (heigth_face / 2), xmin + (width_face / 2)]

    new_ymin = int(center_face[0] - margin)
    new_xmin = int(center_face[1] - margin)

    new_ymax = int(center_face[0] + margin)
    new_xmax = int(center_face[1] + margin)

    face = get_ROI(new_xmin, new_ymin, new_xmax, new_ymax, frame)
    return face


def get_ROI(x1, y1, x2, y2, frame):

    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0

    x2 = x2 if x2 <= frame.shape[1] else frame.shape[1]
    y2 = y2 if y2 <= frame.shape[0] else frame.shape[0]

    return frame[y1:y2, x1:x2]


def check_border_limit(coord, border_limit):
    if coord < 0:
        return 0
    elif coord > border_limit:
        return border_limit
    else:
        return coord


def format_names_to_bottle(list_names):
    import yarp

    list_objects_bottle = yarp.Bottle()
    list_objects_bottle.clear()

    for name in list_names:
        yarp_object_bottle = yarp.Bottle()
        yarp_object_bottle.addString(name[0])
        yarp_object_bottle.addDouble(round(float(name[1]), 2))
        list_objects_bottle.addList().read(yarp_object_bottle)

    return list_objects_bottle


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def get_tensor_from_image( img_path, trans):
        frame = Image.open(img_path)
        tensor = trans(frame).unsqueeze(0).cuda(0)
        return tensor


def survey(df, category):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(df.columns)
    data = np.array(list(np.transpose(df.values)))
    data_cum = data.cumsum(axis=1)

    colors = sns.color_palette("Set1", n_colors=len(labels))
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

    #category_colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots()
    # figsize = (30, 15)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
#     ax.set_xticklabels(fontsize=36)

    for i, (colname, color) in enumerate(zip(category, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b = color
#         text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#         ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(category), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='medium')

    return fig, ax


def format_timestamp_to_date(filename):
    file_name = filename.split(".")[0]
    timestamp = file_name.split("timestamp/")[1]
    if "_" in timestamp:
        timestamp = timestamp.split("_")[0]

    dt = datetime.fromtimestamp(int(timestamp))
    if dt.year == 2022:
        final_date_code = dt.strftime("%d/%m/%Y")
        return final_date_code
    else:
        return None


# create and save embeddings
def create_embeddings(encoder, path):

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    dirs = os.listdir(path)

    for people_dir in dirs:
        for subfolder in os.listdir(os.path.join(path, people_dir)):
            list_img_filenames = []
            for ext in ('*.png', '*.jpg'):
                list_img_filenames.extend(glob.glob(os.path.join(path, people_dir, subfolder, ext)))

            for i, img_path in enumerate(list_img_filenames):
                img_name = os.path.basename(img_path).split('.')[0]

                if not os.path.exists(os.path.join(path, people_dir, subfolder, (img_name + '.npy'))):
                    input_tensor = get_tensor_from_image(img_path, trans)
                    embeddings = encoder(input_tensor).data.cpu().numpy()
                    enbeddings_path = os.path.join(path, people_dir, subfolder) + f"/{img_name}_emb.npy"
                    np.save(enbeddings_path, embeddings.ravel())


def create_embeddings_from_list(model, list_of_paths, saving_folder):

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization,
    ])

    for file in list_of_paths:

        file_name = os.path.basename(file[0].split('.')[0])
        file_root = file[0].split(file_name)[0]

        if not os.path.exists(os.path.join(file_root, saving_folder)):
            os.makedirs(os.path.join(file_root, saving_folder))

        if not os.path.exists(os.path.join(file_root, saving_folder, file_name + '_emb_new.npy')):
            input_tensor = get_tensor_from_image(file[0], trans)
            embeddings = model(input_tensor).data.cpu().numpy()
            enbeddings_path = os.path.join(file_root, saving_folder, file_name + '_emb_new.npy')
            np.save(enbeddings_path, embeddings.ravel())


def compute_dataframe(root_path, df):
    dates_folder = []
    for folder in os.listdir(root_path):
        counter = 0
        for subfolder in os.listdir(os.path.join(root_path, folder)):
            if subfolder != 'timestamp':
                counter = 0
                date = subfolder[:10]
                date = datetime.strptime(date, "%Y_%m_%d").strftime('%d/%m/%Y')
                counter = len(glob.glob1(os.path.join(root_path, folder, subfolder), "*.png")) + len(glob.glob1(os.path.join(root_path, folder, subfolder), "*.jpg"))

                if date not in dates_folder:
                    dates_folder.append(date)
                    df.append(pd.Series(name=date, dtype='datetime64[ns]'))
                    df.at[date, folder] = counter
                else:
                    val = df.at[date, folder]
                    val = 0 if pd.isna(val) else val
                    df.at[date, folder] = val + counter


def flatten(t):
    return [item for sublist in t for item in sublist]


def area_under_gaussian_at_left(t, mu, sigma):
    a = t - mu
    b = np.sqrt(2) * sigma
    return .5 * (1 + erf(a / b))


def area_under_gaussian_at_right(t, mu, sigma):
    return 1 - self.area_under_gaussian_at_left(t, mu, sigma)