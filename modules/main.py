import sys
import os
import time
import scipy
import dlib
import cv2 as cv
import numpy as np

import yarp

from torchvision import transforms
import torch, torchaudio
from faceRecognition.utils import *
from modules.DatabaseHandler import DatabaseHandler
from modules.EmbeddingsHandler import EmbeddingsHandler
from modules.WorkingMemoryHandler import WorkingMemoryHandler


def info(msg):
    print("[INFO] {}".format(msg))

class PersonsRecognition(yarp.RFModule):
    """
    Description:
        Class to recognize a person from the audio or the face (for now only face)
    Args:
        input_port  : Audio from remoteInterface, raw image from iCub cameras
    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        self.device = None

        # handle port for the RFModule
        self.module_name = None
        self.handle_port = None
        self.process = False

        # Ports

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Other ports
        self.label_outputPort = yarp.Port()
        self.eventPort = yarp.BufferedPortBottle()
        # self.eventPort = None
        # self.label_outputPort = None

        # Define port to receive an Image
        self.image_in_port = yarp.BufferedPortImageRgb()
        self.face_coord_port = yarp.BufferedPortBottle()

        # Port to query and update the memory (OPC)
        self.opc_port = yarp.RpcClient()


        # Module parameters
        self.predictions = []
        self.database = None
        self.dataset_path = None
        self.length_input = None
        self.resample_trans = None
        self.speaker_emb = []
        self.name = ""

        # Image parameters
        self.width_img = None
        self.height_img = None
        self.input_img_array = None
        self.frame = None
        self.coord_face = None
        self.threshold_face = None
        self.face_emb = []

        # Model face recognition
        self.model_face = None
        self.db_embeddings_face = None
        self.trans = None
        self.faces_img = []
        self.face_coord_request = None
        self.face_model_path = None

        # RPC commands
        self.save_face = False
        self.predict = False

    def configure(self, rf):

        success = True

        # Module parameters
        self.module_name = rf.check("name",
                                    yarp.Value("PersonRecognition"),
                                    "module name (string)").asString()

        # Root path of the embeddings database (voice & face) (string)
        self.dataset_path = rf.find("dataset_path").asString()

        print("dataset path is", self.dataset_path)
        self.database = DatabaseHandler(self.dataset_path)

        # Threshold for detection (double)
        self.threshold_face = rf.find("threshold_face").asDouble()

        # Path of the model for face embeddings (string)"
        self.face_model_path = rf.find("face_model_path").asString()

        # Width and height of the input image
        self.width_img = rf.find('width_img').asInt()
        self.height_img = rf.find('height_img').asInt()

        # Opening Ports
        self.handle_port.open('/' + self.module_name)

        #todo: check event and labels ports. are they necessary?
        self.eventPort.open('/' + self.module_name + '/events:i')
        self.label_outputPort.open('/' + self.module_name + '/label:o')

        # port to receive the images from robot camera
        self.image_in_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        # port to receive the coordinates from MOT
        self.face_coord_port.open('/' + self.module_name + '/coord:i')
        self.face_coord_port.setStrict(False)

        # port to communicate with working memory (OPC)
        self.opc_port.open('/' + self.module_name + '/OPC:rpc')

        self.working_memory_handler = WorkingMemoryHandler(self.opc_port)

        # Set the device for inference for the models
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        success &= self.load_model_face()
        info("Initialization complete")
        return success

    def interruptModule(self):
        print("[INFO] Stopping the modules")
        self.audio_in_port.interrupt()
        self.label_outputPort.interrupt()
        self.eventPort.interrupt()
        self.handle_port.interrupt()
        self.image_in_port.interrupt()
        self.face_coord_port.interrupt()

        return True

    def close(self):
        print("[INFO] Closing the modules")
        self.audio_in_port.close()
        self.handle_port.close()
        self.label_outputPort.close()
        self.image_in_port.close()
        self.eventPort.close()
        self.face_coord_port.close()

        return True

    def load_model_face(self):
        try:
            self.model_face = torch.load(self.face_model_path)
            self.model_face.eval()

            self.db_embeddings_face = EmbeddingsHandler(os.path.join(self.dataset_path, "face"), threshold=self.threshold_face)

            # Transform for  face embeddings
            self.trans = transforms.Compose([
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization,
                transforms.Resize((180, 180))
            ])

        except FileNotFoundError:
            info(f"Unable to find dataset {EmbeddingsHandler(os.path.join(self.dataset_path, 'face'))} \
            or model {self.face_model_path}")
            return False

        return True

    def respond(self, command, reply):

        #todo: add help command
        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "start":
            reply.addString("ok")
            self.process = True

        elif command.get(0).asString() == "predict":
            self.predict = True
            reply.addString("ok")

        elif command.get(0).asString() == "stop":
            self.process = False
            reply.addString("ok")

        elif command.get(0).asString() == "stoppredict":
                self.predict = False
                reply.addString("ok")

        #todo: check this function
        elif command.get(0).asString() == "check":
            if command.get(1).asString() == "tracker":
                new_detection = []
                new_detection.append(command.get(2).asList().get(0).asDouble())
                new_detection.append(command.get(2).asList().get(1).asDouble())
                new_detection.append(command.get(2).asList().get(2).asDouble())
                new_detection.append(command.get(2).asList().get(3).asDouble())

                name_to_assign, id_to_assign = self.check_existing_face(new_detection)
                if name_to_assign:
                    reply.addString(name_to_assign)
                    reply.addString(id_to_assign)
                else:
                    reply.addString("nack")

        elif command.get(0).asString() == "save":
            if command.get(1).asString() == "face":
                self.save_face = True
            else:
                name = command.get(1).asString()
                print("name", name)
                if name in self.db_embeddings_face.data_dict.keys():
                    self.db_embeddings_face.data_dict[name] = self.db_embeddings_face.data_dict[name] + self.face_emb
                else:
                    self.db_embeddings_face.data_dict[name] = self.face_emb

                self.database.save_faces(self.faces_img, self.face_emb, name)
                self.save_face = False
                self.faces_img = []
                self.face_emb = []

            reply.addString("ok")

        elif command.get(0).asString() == "reset":
            self.db_embeddings_face.excluded_entities = []

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == "thr":
                if command.get(2).asString() == "audio":
                    self.threshold_audio = command.get(3).asDouble()
                    self.db_embeddings_audio.threshold = self.threshold_audio
                    reply.addString("ok")
                elif command.get(2).asString() == "face":
                    self.threshold_face = command.get(3).asDouble() if command.get(3).asDouble() > 0 else self.threshold_face
                    self.db_embeddings_face.threshold = self.threshold_face
                    reply.addString("ok")
                else:
                    reply.addString("nack")
            else:
                reply.addString("nack")


        elif command.get(0).asString() == "get":
            if command.get(1).asString() == "thr":
                if command.get(2).asString() == "audio":
                    reply.addDouble(self.threshold_audio)
                elif command.get(2).asString() == "face":
                    reply.addDouble(self.threshold_face)
                else:
                    reply.addString("nack")
            elif command.get(1).asString() == "face":
                self.face_coord_request = [command.get(2).asDouble(), command.get(3).asDouble(), command.get(4).asDouble(),
                                           command.get(5).asDouble()]
                reply.addString("ok")
            else:
                reply.addString("nack")
        else:
            reply.addString("nack")

        return True

    def getPeriod(self):
        """
           Module refresh rate.
           Returns : The period of the module in seconds.
        """
        return 0.05

    def read_image(self):
        input_yarp_image = self.image_in_port.read(False)
        if input_yarp_image:
            input_yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
            self.frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3)).copy()
            return True
        return False

    def get_face_coordinate(self):
        if self.face_coord_port.getInputCount():
            self.coord_face = self.face_coord_port.read(False)
            return self.coord_face is not None

        self.coord_face = None
        return False

    def check_existing_face(self, detection):
        users_to_verify, id_to_verify = self.working_memory_handler.get_name_to_verify()
        face_name = ""
        face_id = ""
        if len(users_to_verify) > 0:
            face_img_list = []
            face_img = face_alignement(detection, self.frame)
            face_img_list.append(face_img)
            current_face_emb = self.get_face_embeddings(face_img_list)

            if len(current_face_emb):
                distances = []
                names = []
                ids = []

                current_face_emb = current_face_emb[0]
                for (user, id) in zip(users_to_verify, id_to_verify):
                    # if user exist in db_embedding folder
                    distances.append(self.db_embeddings_face.get_distance_from_user(current_face_emb, user))
                    names.append(user)
                    ids.append(id)

                # max similarity is min distance (cosine similarity output [-1,1]
                min_distance_index = np.argmax(distances)
                face_name = names[min_distance_index]
                face_id = ids[min_distance_index]

        return face_name, face_id

    def get_face_embeddings(self, images):
        """
       Generate faces embedding from images of faces
       :param images: list of cropped faces (list->np.array)
       :return:  (list->np.array)
        """
        face_embeddings = []
        with torch.no_grad():
            for np_img in images:
                cv.cvtColor(np_img, cv.COLOR_RGB2BGR, np_img)
                input_img = self.trans(np_img)
                input_img = input_img.unsqueeze_(0)
                input = input_img.to(self.device)
                emb = self.model_face(input)
                face_embeddings.append(emb.cpu())

        return face_embeddings

    def predict_face(self, embeddings):
        predicted_faces = []
        score_faces = []
        for emb in embeddings:
            score, face_name = self.db_embeddings_face.get_speaker_db_scan(emb)
            if score == -1:
                face_name = "unknown"

            predicted_faces.append(face_name)
            score_faces.append(score)
        self.db_embeddings_face.excluded_entities = []
        return predicted_faces, score_faces


    def write_label(self, name_speaker, score, mode):
        if self.label_outputPort.getOutputCount():
            name_bottle = yarp.Bottle()
            name_bottle.clear()
            name_bottle.addString(name_speaker)
            name_bottle.addFloat32(score)
            name_bottle.addInt(mode)

            self.label_outputPort.write(name_bottle)

    #TODO: move to utils
    def format_name(self, name):
        name.strip()
        return name


    def updateModule(self):
        current_face_emb = []
        current_id_faces = []
        person_name, audio_score = "unknown", 0

        record_image = self.read_image()

        self.working_memory_handler.get_name_in_memory()
        self.get_face_coordinate()

        if self.process:

            if record_image and self.frame.size != 0 and self.coord_face:
                try:
                    current_id_faces, self.coord_face = format_face_coord(self.coord_face)
                    face_img = [face_alignement(f, self.frame) for f in self.coord_face]
                    current_face_emb = self.get_face_embeddings(face_img)

                    if self.save_face and len(current_face_emb) > 0:

                        self.faces_img = self.faces_img + face_img
                        self.face_emb.append(current_face_emb[0].numpy())

                except Exception as e:
                    info("Exception while computing face embeddings" + str(e))

            # If we have a face embedding
            if self.predict:
                if person_name != 'unknown' and len(current_face_emb):
                    info("Got Face embeddings")
                    faces_name, face_scores = self.predict_face(current_face_emb)

                    unknown_faces = []
                    distances = []
                    for face_id, emb, name, score in zip(current_id_faces, current_face_emb, faces_name, face_scores):
                        if name != "unknown":
                            name = self.format_name(name)
                            self.working_memory_handler.set_name_memory(face_id, name)
                            print("Predicted for face_id {} : {} with score {}".format(face_id, name, score))
                        else:
                            distances.append(self.db_embeddings_face.get_distance_from_user(emb, person_name))
                            unknown_faces.append(face_id)

                    if len(unknown_faces):
                        min_distance_index = np.argmax(distances)
                        min_face_id = unknown_faces.pop(min_distance_index)
                        self.working_memory_handler.set_name_memory(min_face_id, person_name)
                        # print("Speaker name closest to unknown face is {} ".format(speaker_name))

                        for face_id in unknown_faces:
                            self.working_memory_handler.set_name_memory(face_id, "unknown")

                elif len(current_face_emb):
                    faces_name, scores = self.predict_face(current_face_emb)
                    for face_id, name, score in zip(current_id_faces, faces_name, scores):
                        self.working_memory_handler.set_name_memory(face_id, name)
                        print("Predicted for face_id {} : {} with score {}".format(face_id, name, score))
                else:
                    pass


        return True




if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        info("Unable to find a yarp server. Exiting ...")
        sys.exit(1)

    yarp.Network.init()

    # Instantiate Module
    person_recognition = PersonsRecognition()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('person-recognition-HRI')  #person-recognition-HRI, where to find .ini files for this application
    rf.setDefaultConfigFile('PersonRecognition.ini')  # name of the default .ini file NO, NAME OF THE FOLDER

    if rf.configure(sys.argv):
        person_recognition.runModule(rf)

    person_recognition.close()
    sys.exit()
