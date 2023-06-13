import os
import yarp

class WorkingMemoryHandler(object):
    """
    Handle the communication with and modification of Working Memory (OPC)
    """

    def __init__(self, opc_port):
        self.opc_port = opc_port

        # TODO: move to working_memory utils

    def set_name_memory(self, face_id, face_name):
        if self.opc_port.getOutputCount():
            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("id_tracker")
            cond1.addString("==")
            cond1.addString(face_id)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            if list_id.size():
                cmd = yarp.Bottle()

                cmd.addString("get")
                list_all = cmd.addList()
                list_1 = list_all.addList()
                list_1.addString("id")
                list_1.addInt(list_id.get(0).asInt())

                list_2 = list_all.addList()
                list_2.addString("propSet")
                list_3 = list_2.addList()
                list_3.addString("verified")

                reply_ver = yarp.Bottle()

                self.opc_port.write(cmd, reply_ver)
                print("Sent cmd to OPC {}, and received response {}".format(cmd.toString(), reply_ver.toString()))

                verified = reply_ver.get(1).asList().get(0).asList().get(1).asInt()
                if verified == 0:
                    reply2 = yarp.Bottle()
                    cmd = yarp.Bottle()
                    cmd.addString("set")
                    list_cmd = cmd.addList()
                    id_cmd = list_cmd.addList()
                    id_cmd.addString("id")
                    id_cmd.addInt(list_id.get(0).asInt())
                    label_cmd = list_cmd.addList()
                    label_cmd.addString("label_tracker")
                    label_cmd.addString(face_name.strip())
                    # cmd_str = "set ((id " + str(list_id.get(0).asInt()) + ") (label_tracker" + face_name + "))"

                    self.opc_port.write(cmd, reply2)
                    print("Sent cmd to OPC {} and received reply {}".format(cmd.toString(), reply2.toString()))
                    return "ack" + reply2.get(0).asString()
        return False


    def get_name_in_memory(self):
        if self.opc_port.getOutputCount():

            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("verified")
            cond1.addString("==")
            cond1.addInt(1)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            for i in range(list_id.size()):
                cmd_str = "get ((id " + str(list_id.get(i).asInt()) + ") (propSet (label_tracker)))"
                cmd = yarp.Bottle(cmd_str)
                reply_id = yarp.Bottle()
                self.opc_port.write(cmd, reply_id)
                if reply_id.size() > 0:
                    name = reply_id.get(1).asList().get(0).asList().get(1).asString()
                    self.db_embeddings_face.excluded_entities.append(name)
                    self.db_embeddings_audio.excluded_entities.append(name)
        return False


    def get_name_to_verify(self):
        if self.opc_port.getOutputCount():

            reply = yarp.Bottle()
            cmd = yarp.Bottle("ask")
            list_condition = cmd.addList()
            cond1 = list_condition.addList()
            cond1.addString("verified")
            cond1.addString("==")
            cond1.addInt(1)
            list_condition.addString("&&")
            cond2 = list_condition.addList()
            cond2.addString("active")
            cond2.addString("==")
            cond2.addInt(0)

            self.opc_port.write(cmd, reply)
            list_id = reply.get(1).asList().get(1).asList()

            name_to_verify = []
            id_to_verify = []
            if list_id.size() > 0:
                reply_id = yarp.Bottle()
                for i in range(list_id.size()):
                    cmd_str = "get ((id " + str(list_id.get(i).asInt()) + ") (propSet (label_tracker id_tracker)))"
                    cmd = yarp.Bottle(cmd_str)
                    self.opc_port.write(cmd, reply_id)
                    name = reply_id.get(1).asList().get(1).asList().get(1).asString()
                    id = reply_id.get(1).asList().get(0).asList().get(1).asString()
                    name_to_verify.append(name)
                    id_to_verify.append(id)

            return name_to_verify, id_to_verify

        return False