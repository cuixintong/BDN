from .FFA import *
from .DCP import *
from .Jx import *
from .MVSA import *

class BDN(nn.Module):
    def __init__(self, gps, blocks, conv = default_conv):
        super(BDN, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        self.FFA = FFANet(gps, blocks, conv)
        self.MVSA = MVSA(3,3)
        # self.DCP = DCP()
        self.Jx = Jx(3,3)
    def forward(self,x,x_dcp,x2 = 0, Val=False):
        # x_dcp = x.detach().cpu().numpy()
        # x_dcp = DCP(x_dcp)
        # x_dcp = DCP(file_path)
        # x_dcp = torch.from_numpy(x_dcp)
        out_J = self.Jx(x_dcp)
        _,out_T = self.FFA(x,x_dcp)

        # 111

        if Val == False:
            out_A, _ = self.MVSA(x, x_dcp)  # out_A 需要输出[1,3,1,1]
        # else:
        #     out_A = self.ANet(x2)



        # out_T = {t.tolist for t in out_T}
        # out_A = {a.tolist for a in out_A}
        # out_J = {j.tolist for j in out_J}

        out_I = out_T * out_J + (1 - out_T) * out_A

        return out_J, out_T, out_A, out_I





