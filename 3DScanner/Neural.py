import numpy
import numpy as np
import onnxruntime as ort

model_final_cnn = "history2/1_point_cloud_model_final_309630_wd_200_700.onnx"
model_enc1_cnn = "history2/1_point_cloud_model_enc1_309630_wd_200_700.onnx"
model_enc2_cnn = "history2/1_point_cloud_model_enc2_309630_wd_200_700.onnx"

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
