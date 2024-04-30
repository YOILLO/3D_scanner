import random

import numpy
import numpy as np
import onnxruntime as ort
import torch
import cProfile
import re
import pstats, math
import io
import open3d as o3d
from pycpd import RigidRegistration
from torchviz import make_dot

model_final = "onnx/0_point_cloud_model_final_50_wd_20_200.onnx"
model_enc1 = "onnx/0_point_cloud_model_enc1_50_wd_20.onnx"
model_enc2 = "onnx/0_point_cloud_model_enc2_50_wd_20.onnx"

class onnxNN():
    def __init__(self, decoder_width):
        self.session_enc1 = ort.InferenceSession(model_enc1)
        self.session_enc2 = ort.InferenceSession(model_enc2)
        self.session_final = ort.InferenceSession(model_final)
        self.decoder_with = decoder_width

    def forward(self, src, dst):
        tmp1 = [np.array([0.0] * self.decoder_with, dtype=numpy.single)]
        for i in src:
            tmp1 = self.session_enc1.run(None, {"tmp": np.array(tmp1[0], dtype=numpy.single), "point": np.array(i, dtype=numpy.single)})

        tmp2 = [np.array([0.0] * self.decoder_with, dtype=numpy.single)]
        for i in dst:
            tmp2 = self.session_enc2.run(None, {"tmp": np.array(tmp2[0], dtype=numpy.single), "point": np.array(i, dtype=numpy.single)})

        return self.session_final.run(None, {"src": tmp1[0], "tgt": tmp2[0]})

model_final_cnn = "onnx/1_point_cloud_model_final_19500_wd_40_200.onnx"
model_enc1_cnn = "onnx/1_point_cloud_model_enc1_19500_wd_40_200.onnx"
model_enc2_cnn = "onnx/1_point_cloud_model_enc2_19500_wd_40_200.onnx"

class onnxCNN():
    def __init__(self, decoder_width, length):
        self.session_enc1 = ort.InferenceSession(model_enc1_cnn)
        self.session_enc2 = ort.InferenceSession(model_enc2_cnn)
        self.session_final = ort.InferenceSession(model_final_cnn)
        self.decoder_with = decoder_width
        self.decoder_lengh = length

    def forward(self, src, dst):
        tmp1 = [np.array([0.0] * self.decoder_with, dtype=numpy.single)]
        for i in range(len(src)//self.decoder_lengh):
            tmp1 = self.session_enc1.run(None, {"tmp": np.array(tmp1[0], dtype=numpy.single),
                                                "point": np.array([src[i*self.decoder_lengh:(i + 1)*self.decoder_lengh]],
                                                                  dtype=numpy.single)})

        tmp2 = [np.array([0.0] * self.decoder_with, dtype=numpy.single)]
        for i in range(len(dst)//self.decoder_lengh):
            tmp2 = self.session_enc2.run(None, {"tmp": np.array(tmp2[0], dtype=numpy.single),
                                                "point": np.array([dst[i*self.decoder_lengh:(i + 1)*self.decoder_lengh]],
                                                                  dtype=numpy.single)})

        return self.session_final.run(None, {"src": tmp1[0], "tgt": tmp2[0]})


model = onnxNN(20)


def get_rand():
    return (random.random() - 0.5) * 2 * 1000


# arr1 = [[get_rand(), get_rand(), get_rand()]]
# arr2 = [[get_rand(), get_rand(), get_rand()]]
#
# nn_file = open("onnx/nn.log", "w")
#
# for i in range(10_000):
#     pr = cProfile.Profile()
#
#     arr1.append([get_rand(), get_rand(), get_rand()])
#     arr2.append([get_rand(), get_rand(), get_rand()])
#
#     pr.enable()
#     model.forward(np.array(arr1, dtype=float), np.array(arr2, dtype=float))
#     pr.disable()
#
#     result = io.StringIO()
#     pstats.Stats(pr, stream=result).print_stats()
#     result = result.getvalue()
#
#     resultArr = result.splitlines()
#
#     nn_file.write(str(len(arr1)))
#     nn_file.write(resultArr[0] + "\n")
#
#     print(len(arr1), resultArr[0])
#
# nn_file.close()
#
# model_cnn = onnxCNN(40, 200)
#
# arr1cnn = [[get_rand(), get_rand(), get_rand()]]
# arr2cnn = [[get_rand(), get_rand(), get_rand()]]
#
# cnn_file = open("onnx/cnn.log", "w")
#
# for i in range(10_000):
#     pr = cProfile.Profile()
#
#     arr1cnn.append([get_rand(), get_rand(), get_rand()])
#     arr2cnn.append([get_rand(), get_rand(), get_rand()])
#
#     arr1tmp = arr1cnn.copy()
#     arr2tmp = arr1cnn.copy()
#
#     while len(arr1tmp) % 200:
#         arr1tmp.append([0, 0, 0])
## arr1 = [[get_rand(), get_rand(), get_rand()]]
# arr2 = [[get_rand(), get_rand(), get_rand()]]
#
# nn_file = open("onnx/nn.log", "w")
#
# for i in range(10_000):
#     pr = cProfile.Profile()
#
#     arr1.append([get_rand(), get_rand(), get_rand()])
#     arr2.append([get_rand(), get_rand(), get_rand()])
#
#     pr.enable()
#     model.forward(np.array(arr1, dtype=float), np.array(arr2, dtype=float))
#     pr.disable()
#
#     result = io.StringIO()
#     pstats.Stats(pr, stream=result).print_stats()
#     result = result.getvalue()
#
#     resultArr = result.splitlines()
#
#     nn_file.write(str(len(arr1)))
#     nn_file.write(resultArr[0] + "\n")
#
#     print(len(arr1), resultArr[0])
#
# nn_file.close()
#
# model_cnn = onnxCNN(40, 200)
#
# arr1cnn = [[get_rand(), get_rand(), get_rand()]]
# arr2cnn = [[get_rand(), get_rand(), get_rand()]]
#
# cnn_file = open("onnx/cnn.log", "w")
#
# for i in range(10_000):
#     pr = cProfile.Profile()
#
#     arr1cnn.append([get_rand(), get_rand(), get_rand()])
#     arr2cnn.append([get_rand(), get_rand(), get_rand()])
#
#     arr1tmp = arr1cnn.copy()
#     arr2tmp = arr1cnn.copy()
#
#     while len(arr1tmp) % 200:
#         arr1tmp.append([0, 0, 0])
#
#     while len(arr2tmp) % 200:
#         arr2tmp.append([0, 0, 0])
#
#     pr.enable()
#     model_cnn.forward(np.array(arr1tmp, dtype=float), np.array(arr2tmp, dtype=float))
#     pr.disable()
#
#     result = io.StringIO()
#     pstats.Stats(pr, stream=result).print_stats()
#     result = result.getvalue()
#
#     resultArr = result.splitlines()
#
#     cnn_file.write(str(len(arr1cnn)))
#     cnn_file.write(resultArr[0] + "\n")
#
#     print(len(arr1cnn), resultArr[0])
#

#     while len(arr2tmp) % 200:
#         arr2tmp.append([0, 0, 0])
#
#     pr.enable()
#     model_cnn.forward(np.array(arr1tmp, dtype=float), np.array(arr2tmp, dtype=float))
#     pr.disable()
#
#     result = io.StringIO()
#     pstats.Stats(pr, stream=result).print_stats()
#     result = result.getvalue()
#
#     resultArr = result.splitlines()
#
#     cnn_file.write(str(len(arr1cnn)))
#     cnn_file.write(resultArr[0] + "\n")
#
#     print(len(arr1cnn), resultArr[0])
#

arr1 = [[get_rand(), get_rand(), get_rand()]]
arr2 = [[get_rand(), get_rand(), get_rand()]]

threshold = 0.02
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

nn_file = open("onnx/pp.log", "w")

for i in range(10_000):
    pr = cProfile.Profile()

    arr1.append([get_rand(), get_rand(), get_rand()])
    arr2.append([get_rand(), get_rand(), get_rand()])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(np.array(arr1))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.array(arr2))

    pr.enable()
    o3d.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    pr.disable()

    result = io.StringIO()
    pstats.Stats(pr, stream=result).print_stats()
    result = result.getvalue()

    resultArr = result.splitlines()

    nn_file.write(str(len(arr1)))
    nn_file.write(resultArr[0] + "\n")

    print(len(arr1), resultArr[0])
