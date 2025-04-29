# import math
# import torch
# from .ciphertext import Plaintext, Cipher, PreEncodeValues

# def get_item(item_name, content_map):
#     if item_name in content_map:
#         return content_map[item_name]
#     return None

# class BsContext:
#     def __init__(self, content_map):
#         self.m_U0PreFFT_ = get_item("m_U0PreFFT", content_map)
#         self.m_U0hatTPreFFT_ = get_item("m_U0hatTPreFFT", content_map)

#         if isinstance(self.m_U0PreFFT_[0][0], Plaintext):
#             for i in range(len(self.m_U0hatTPreFFT_)):
#                 for j in range(len(self.m_U0hatTPreFFT_[i])):
#                     self.m_U0hatTPreFFT_[i][j].cv = [torch.tensor(self.m_U0hatTPreFFT_[i][j].cv, dtype = torch.uint64)]
#                     Cipher._id_counter = max(Cipher._id_counter, self.m_U0hatTPreFFT_[i][j].cipher_id)

#             for i in range(len(self.m_U0PreFFT_)):
#                 for j in range(len(self.m_U0PreFFT_[i])):
#                     self.m_U0PreFFT_[i][j].cv = [torch.tensor(self.m_U0PreFFT_[i][j].cv, dtype = torch.uint64)]
#                     Cipher._id_counter = max(Cipher._id_counter, self.m_U0PreFFT_[i][j].cipher_id)
#         elif isinstance(self.m_U0PreFFT_[0][0], PreEncodeValues):
#             for i in range(len(self.m_U0hatTPreFFT_)):
#                 for j in range(len(self.m_U0hatTPreFFT_[i])):
#                     self.m_U0hatTPreFFT_[i][j].encoded_values = torch.tensor(self.m_U0hatTPreFFT_[i][j].encoded_values)

#             for i in range(len(self.m_U0PreFFT_)):
#                 for j in range(len(self.m_U0PreFFT_[i])):
#                     self.m_U0PreFFT_[i][j].encoded_values = torch.tensor(self.m_U0PreFFT_[i][j].encoded_values)
        
#         self.BS_FFT = {}
#         for i in range(len(self.m_U0hatTPreFFT_)):
#             for j in range(len(self.m_U0hatTPreFFT_[i])):
#                 self.BS_FFT["{}_{}_{}".format("C2S", i, j)] = self.m_U0hatTPreFFT_[i][j]

#         for i in range(len(self.m_U0PreFFT_)):
#             for j in range(len(self.m_U0PreFFT_[i])):
#                 self.BS_FFT["{}_{}_{}".format("S2C", i, j)] = self.m_U0PreFFT_[i][j]

#     def to_cuda(self):
#         for key, value in self.BS_FFT.items():
#             if isinstance(value, Plaintext):
#                 self.BS_FFT[key].cv = [self.BS_FFT[key].cv[0].cuda()]
#             elif isinstance(value, PreEncodeValues):
#                 self.BS_FFT[key].encoded_values = self.BS_FFT[key].encoded_values.cuda()
#             else:
#                 raise TypeError("Unsupported type for BS_FFT value: {}".format(type(value)))
