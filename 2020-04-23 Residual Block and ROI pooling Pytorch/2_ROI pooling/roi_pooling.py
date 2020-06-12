import numpy as np
import random
import torch
import torchvision.ops.roi_pool as roi_pool


# Implement a ROI Pooling operator. Your code will be given the following variables:

#     input, a mini-batch of feature maps (a torch.Tensor with shape (n, C, H, W) and dtype torch.float32)
#     boxes, a list of bounding box coordinates on which you need to perform the ROI Pooling. boxes will be a list of (L,4) torch.Tensor with dtype torch.float32, where boxes[i] will refer to the i-th element of the batch, and contain L coordinates in the format (y1, x1, y2, x2)
#     a tuple of integers output_size, containing the number of cells over which pooling is performed, in the format (heigth, width)

# The code should produce an output torch.Tensor out with dtype torch.float32 and shape (n, L, C, output_size[0], output_size[1]).

# https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension
# http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

N = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)

input = torch.rand(N, C, H, W)  # N:2 C:16 H:5 W:7
print(
    "input n:{} C:{} H:{} W:{}".format(N, C, H, W)
)  # 2 input, each in input has 16 channels, 5 Height and 7 Width

boxes = [
    torch.zeros(L, 4) for _ in range(N)  # fill a list of tensor
]  # L:5 coordinates:4 n:2
# for each input (n=2) I have 5 regions/rectangles with 4 coordinates that #define bounding box on one of the input
print("boxes L:{} coordinates:4 N:{}".format(L, N))
print("boxes ", boxes[0].shape)
for i in range(N):  # coordinates are random but valid! so stay in the sample
    boxes[i][:, 0] = torch.rand(L) * (H - oH)  # y
    boxes[i][:, 1] = torch.rand(L) * (W - oW)  # x
    boxes[i][:, 2] = oH + torch.rand(L) * (H - oH)  # w
    boxes[i][:, 3] = oW + torch.rand(L) * (W - oW)  # h

    boxes[i][:, 2:] += boxes[i][:, :2]
    boxes[i][:, 2] = torch.clamp(boxes[i][:, 2], max=H - 1)
    boxes[i][:, 3] = torch.clamp(boxes[i][:, 3], max=W - 1)

output_size = (oH, oW)  # oH:4 oW3
# I want final size after pooling is 4x3: each box with 4 coordinated is divided into a 4x3 grid, on each grid I take the max in that grid -> output is 4x3 tensor which will be stacked for each channel and for each sample
print("output_size oH:{} oW{}".format(oH, oW))

out = torch.zeros(N, L, C, oH, oW)  # n:2 L:5 C:16 oH:4 oW:3
# for each input sample and relative channels I have 5 tensor each one with a 4x3 grid after the pooling -> so final size is fixed for any bounding boxes!
print("output N:{} L:{} C:{} oH:{} oW:{}".format(N, L, C, oH, oW))
out = out.to(torch.float32)  # convert cast change type tensor
out = out.long()  # long == int64

(N, C, H, W) = input.shape
(L, _) = boxes[0].shape
(oH, oW) = output_size[0], output_size[1]


def get_indexes(i, j, y1, x1, y2, x2, oH, oW):
    out = []
    y_start = torch.floor(y1 + i * (y2 - y1 + 1) / oH)
    out.append(y_start)  # concat a list of tensor
    y_end = torch.ceil(y1 + (i + 1) * (y2 - y1 + 1) / oH)
    out.append(y_end)
    x_start = torch.floor(x1 + j * (x2 - x1 + 1) / oW)
    out.append(x_start)
    x_end = torch.ceil(x1 + (j + 1) * (x2 - x1 + 1) / oW)
    out.append(x_end)
    out = [x.type(torch.int32) for x in out]  # change cast type list tensor
    # print [x.data for x in out] other method
    return out


for n in range(N):  # for each sample in the batch
    for l in range(L):  # for each bounding box of that sample
        for i in range(oH):  # for each row position in the final tensor
            # print(boxes[n])
            for j in range(oW):  # for each col position in the final tensor
                # print(boxes[n][l])
                (y1, x1, y2, x2) = boxes[n][l]
                # get for that sample the five bounding box assigned (l=0 -> 4)
                (y1, x1, y2, x2) = torch.round(
                    boxes[n][l]
                )  # round to nearest int those values
                # print(torch.round(boxes[n][l]))
                (y_start, y_end, x_start, x_end) = get_indexes(
                    i, j, y1, x1, y2, x2, oH, oW,
                )  # (i,j) define the number and the position of the 4x3 grid
                # so for each sample for each bounding box I do this 4x3= 12 times cause i need 12 sectors over which apply the max and get a 4x3 final matrix
                # IMPORTANT: REGIONS CAN OVERLAP! IT DOES NOT HAPPEN IF I SUBSTITUTE .floor and .ceil with .round!

                slice = input[
                    n, :, y_start:y_end, x_start:x_end
                ]  # input is torch.Size([2, 16, 5, 7]) -> I take 1 sample from the 0 one to the n-1, all channels and a portion defined by those 12 group of coordinates -> torch.Size([16, 2, 2])
                slice, _ = torch.max(
                    torch.max(slice, dim=1)[0], dim=1
                )  # IMPORTANT PART: from this tensor, so a 2x2 image with 16 channels I want to take max value in each of the 12 grid (in this case grid are only 2x2 cause it's too little) along row and along col -> so in numpy would be something like np.amax(slice, axis=(-1, -2)), here you have to do 2 times taking the max over axis=1 -> torch.Size([16])
                # inner torch.max torch.Size([16, 2, 2]) -> torch.Size([16, 2])
                # outer torch.max torch.Size([16, 2) -> torch.Size([16])
                # torch.max return tensor after max and indexes of max values, so discard second value
                out[n, l, :, i, j] = slice  # now you can insert a whole channel in ':'


out_pytorch = roi_pool(input, boxes, (oH, oW), spatial_scale=1.0)
out_pytorch = out_pytorch.reshape((N, L, C, oH, oW))
# Computes element-wise equality
# print(torch.eq(out, out_pytorch, out=None))  # not equal


# slice = input.numpy()[:, :, 3:6, 0:4] convert to numpy
# print (slice.data) get tensor data
