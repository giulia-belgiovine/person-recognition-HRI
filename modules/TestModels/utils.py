# This file ìs temporary. It stores functions used for data analysis of the paper.



def _load_dataset_with_list(self):
    """
    Load the set of embeddings define in the root_dir
    :return:
    """
    for label in self.list_target:
        list_emb = []
        name_emb = []
        same_class_files = [s for s, l in self.list_files if l == label]
        name = self.target_dict[label]

        for f in same_class_files:
            filename = re.split('\.png|\.jpg', f)[0]
            basename = os.path.basename(filename)
            root = filename.split(basename)[0]

            if self.emb_folder:
                emb_filename = os.path.join(root, self.emb_folder, basename + "_emb_new.npy")
            else:
                emb_filename = filename + "_emb.npy"

            if os.path.exists(emb_filename):
                list_emb.append(np.load(emb_filename).squeeze())
                name_emb.append(emb_filename)

        print("Length of embeddings_file for class {} is {}".format(name, len(list_emb)))

        mean = np.array(list_emb).mean(axis=0)
        self.mean_embedding[name] = mean
        # self.filename_dict[name] = name_emb
        self.data_dict[name] = list_emb
        self.name_dict[label] = name
        return


def flatten(self, t):
    return [item for sublist in t for item in sublist]


def area_under_gaussian_at_left(self, t, mu, sigma):
    a = t - mu
    b = np.sqrt(2) * sigma
    return .5 * (1 + erf(a / b))


def area_under_gaussian_at_right(self, t, mu, sigma):
    return 1 - self.area_under_gaussian_at_left(t, mu, sigma)


def load_unknowns(self, path):
    emb_filenames = glob.glob(os.path.join(path, "*.npy"))
    list_emb = [np.load(emb_f).squeeze() for emb_f in emb_filenames]
    list_label = ['unknown'] * len(list_emb)

    return list_emb, list_label


def split_dataset_train_val(self):
    img_list = []
    label_list = []
    for k, v in self.data_dict.items():
        img_list.append(v)
        label_list.append([k]*len(v))

    img_list = self.flatten(img_list)
    label_list = self.flatten(label_list)
    X_train, X_val, y_train, y_val = train_test_split(img_list, label_list, test_size=0.2, stratify=label_list)

    return X_train, X_val, y_train, y_val


def find_best_threshold(self, alpha):
    thresholds_values = np.arange(0.4, 1, 0.05)

    X_train, X_val, Y_train, Y_val = self.split_dataset_train_val()
    list_emb_unk, list_label_unk = self.load_unknowns(os.path.join(self.path_unknown_val, self.emb_folder))

    X_val = X_val + list_emb_unk
    Y_val = Y_val + list_label_unk

    total_loss_all = []
    curr_total_loss = 1

    for thr in thresholds_values:
        FAR_score = 0
        DIR_score = 0

        b1 = len(Y_val) - len(list_label_unk)
        b0 = len(list_label_unk)
        for emb_val, y_val in zip(X_val, Y_val):
            emb_val = torch.from_numpy(emb_val).unsqueeze(0).cuda()
            true_name = y_val

            scores, predicted_name, pos, neg = self.get_speaker_db_scan(emb_val, y_val, X_train, Y_train, thr)
            if scores == -1:
                predicted_name = 'unknown'

            if true_name == 'unknown' and predicted_name != 'unknown':
                FAR_score += 1
            elif true_name != 'unknown' and true_name == predicted_name:
                DIR_score += 1

        DIR = DIR_score/b1
        FAR = FAR_score/b0
        total_loss = alpha * (1-DIR) + (1-alpha) * FAR
        print("with threshold {} DIR is {}, FAR is {}, total_loss is {}".format(thr, DIR, FAR, total_loss))
        total_loss_all.append(total_loss)

        if total_loss < curr_total_loss:
            curr_total_loss = total_loss
        else:
            best_loss_idx = np.argmin(total_loss_all)
            best_threshold = thresholds_values[best_loss_idx]
            print("Best Threshold is ", best_threshold)
            return best_threshold


def find_distances_distribution_thr(self, d, visualize=False):
    file_path = os.path.join('/home/icub/Desktop/logs_IL_analysis/distances/ALL_GALLERY/retrained_check', 'distances_train_{}'.format(d))
    thr80pos = []

    with open(file_path, 'w') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n')
        writer.writerow(["distance", "decision"])
        for person, list_emb in self.data_dict.items():
            for i in range(0, len(list_emb)-1):
                for j in range(i+1, len(list_emb)):
                    emb1 = torch.from_numpy(list_emb[i]).unsqueeze(0).cuda()
                    emb2 = torch.from_numpy(list_emb[j]).unsqueeze(0).cuda()
                    dist = torch.cdist(emb1, emb2).item()
                    row = [dist, "Yes"]
                    writer.writerow(row)

        samples_list = list(self.data_dict.values())

        for i in range(0, len(self.data_dict) - 1):
            for j in range(i + 1, len(self.data_dict)):
                cross_product = itertools.product(samples_list[i], samples_list[j])
                cross_product = list(cross_product)
                for cross_sample in cross_product:
                    emb1 = torch.from_numpy(cross_sample[0]).unsqueeze(0).cuda()
                    emb2 = torch.from_numpy(cross_sample[1]).unsqueeze(0).cuda()
                    dist = torch.cdist(emb1, emb2).item()

                    row = [dist, "No"]
                    writer.writerow(row)

    df = pd.read_csv(file_path)

    tp_mean = df["distance"][df.decision == "Yes"].mean()
    tp_std = df["distance"][df.decision == "Yes"].std()
    # tn_mean = df["distance"][df.decision == "No"].mean()
    # tn_std = df["distance"][df.decision == "No"].std()

    threshold = round(tp_mean + 2 * tp_std, 4)
    thr80pos = 0.8 * threshold

    # max_pos_dist = np.argmax(all_pos_dist)
    #
    # t = np.linspace(np.min(all_pos_dist), np.max(all_neg_dist), 1000)
    # area_neg = self.area_under_gaussian_at_right(t, np.mean(all_neg_dist), np.std(all_neg_dist))
    # area_pos = self.area_under_gaussian_at_left(t, np.mean(all_pos_dist), np.std(all_pos_dist))
    # threshold = t[np.argmax(area_neg + area_pos)]

    # print("Best threshold: {:.2f} --> {:.2f}".format(threshold, thr80pos))

    if visualize:
        df[(df.decision == "No")].distance.plot.hist(color="red", alpha=0.5, bins=100, range=(0, 2),label="Negative Distances")
        df[(df.decision == "Yes")].distance.plot.hist(color="blue", alpha=0.5, bins=100, range=(0, 2),label="Positive Distances")
        plt.title("Train Distances Distribution at Day {}".format(d))
        plt.legend()
        plt.show()

        #self.plot_distances_distribution(all_neg_dist, all_pos_dist, threshold)

    return thr80pos

    # TODO: move in a file for analysis
    def plot_distances_distribution(self, all_neg_dist, all_pos_dist, threshold):
        fig, ax = plt.subplots(1)
        sns.histplot([all_neg_dist, all_pos_dist], binwidth=0.05, label=['neg', 'pos'], ax=ax)
        ax.axvline(threshold, color='black')
        plt.show()


def plot_distances_distribution(self, all_neg_dist, all_pos_dist, threshold):
    fig, ax = plt.subplots(1)
    sns.histplot([all_neg_dist, all_pos_dist], binwidth=0.05, label=['neg', 'pos'], ax=ax)
    ax.axvline(threshold, color='black')
    plt.show()

    def get_distance_from_user(self, emb, identity_to_check):
        max_dist = -1

        if identity_to_check in self.data_dict.keys():
            for person_emb in self.data_dict[identity_to_check]:
                dist = self.similarity_func(torch.from_numpy(person_emb), emb).numpy()
                if dist[0] > max_dist:
                    max_dist = dist[0]

            return max_dist
        return False

    def get_max_distances(self, emb, y_val, thr):
        if thr is None:
            thr = self.threshold
        list_distance = []
        label_list = []
        pos_dist = []
        neg_dist = []

        for speaker_label, list_emb in self.data_dict.items():
            # if speaker_label not in self.excluded_entities:
            for person_emb in list_emb:
                person_emb = torch.from_numpy(person_emb).unsqueeze(0) #.cuda()
                dist = torch.cdist(person_emb, emb).item() #.numpy()
                if y_val == speaker_label:
                   pos_dist.append(dist)
                else:
                   neg_dist.append(dist)
                if dist < thr:
                    list_distance.append(dist)
                    label_list.append(speaker_label)

        return list_distance, label_list


def get_max_distances_train_val(self, emb_val, y_val, X_train, Y_train, thr=None,):
    if thr is None:
        thr = self.threshold
    list_distance = []
    label_list = []
    # emb_val = torch.from_numpy(emb_val).unsqueeze(0).cuda()
    for label, person_emb in zip(Y_train, X_train):
        person_emb = torch.from_numpy(person_emb).unsqueeze(0).cuda()
        dist = torch.cdist(person_emb, emb_val).item()
        if dist < thr:
            list_distance.append(dist)
            label_list.append(label)

    return list_distance, label_list


def get_speaker_db_scan_train(self, emb_val, y_val=[], X_train=[], Y_train=[], thr=None, file_path=[]):
    pos = []
    neg = []
    if X_train:
        distances, labels = self.get_max_distances_train_val(emb_val, y_val, X_train, Y_train, thr)
    else:
        distances, labels = self.get_max_distances(emb_val, y_val, thr)

    if len(distances) == 0:
        return -1, -1, pos, neg

    n = len(distances) if len(distances) < self.n_neighbors else self.n_neighbors
    try:
        max_dist_idx = np.argpartition(distances, -n)[-n:]
        count = dict()
        for i in max_dist_idx:
            if labels[i] not in count.keys():
                count[labels[i]] = [1, distances[i]]
                continue
            count[labels[i]][0] += 1
            count[labels[i]][1] += distances[i]

        max_count = 0
        max_dist = 0
        final_label = ''
        for key, value in count.items():
            if value[0] >= max_count and value[1] >= max_dist:
                max_count = value[0]
                max_dist = value[1]
                final_label = key

        #self.excluded_entities.append(final_label)
        return max_dist/max_count, final_label, pos, neg

    except Exception as e:
        return -1, -1, [], []
