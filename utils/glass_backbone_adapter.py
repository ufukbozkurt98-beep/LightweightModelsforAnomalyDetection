import torch
import torch.nn as nn

# making the dictionary based extractor compatible
class _Tap(nn.Module):  # module that stores the last tensor that passed through it

    def __init__(self):
        super().__init__()  # initializing the base nn.Module class
        self.last = None  # creating the variable to store the last tensor that is seen

    def forward(self, x):  # the input tensor x to the self.last
        self.last = x  # for example self.l1.last = the most recent feature map tensor for layer 1
        return x


# Turns mobilenetv3's dictionary output ({'l1': tensor, 'l2': tensor, 'l3': tensor}) to a backbone that GLASS can
# hook. Because GLASS expects a backbone with named layers for forward hooks. GlassBackboneAdapter passes the
# extracted feature maps through the dummy l1, l2 and l3 modules so that the GLASS can attach and read.
class GlassBackboneAdapter(nn.Module):

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor  # the build_extractor(_) object

        # building the layers that will be hookable for GLASS:
        self.l1 = _Tap()
        self.l2 = _Tap()
        self.l3 = _Tap()

    def forward(self, x):
        feats = self.extractor(x)  # dict: l1,l2,l3
        # pass through tap modules so forward hooks can capture outputs
        self.l1(feats["l1"])  # tap's forward is called
        self.l2(feats["l2"])
        self.l3(feats["l3"])
        return feats["l3"]  # for forward to return something
