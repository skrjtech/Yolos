from __future__ import annotations
from ctypes.wintypes import PBOOL
from dataclasses import dataclass
from typing import Any, Tuple, List, Union
from matplotlib.dates import TU

import torch

from BoundingBox import BoundingBox, BoundingBoxes, BoundingBoxCenter

@dataclass(frozen=True)
class YoloXYBox:
    xmin: float; ymin: float; xmax: float; ymax: float
    labelname: str; labelid: int
    def __call__(self) -> Tuple:
        return (self.xmin, self.ymin, self.xmax, self.ymax, self.labelname, self.labelid)
    def __str__(self) -> str:
        return f"(xmin: {self.xmin:^.3f} | ymin: {self.ymin:^.3f}), (xmax: {self.xmax:^.3f} | ymax: {self.ymax:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"

@dataclass(frozen=True)
class YoloXYCenterIndexBox:
    xi: float; yi: float
    x: float; y: float; w: float; h: float
    labelname: str; labelid: int
    def __call__(self) -> Tuple:
        return (self.xi, self.yi, self.x, self.y, self.w, self.h, self.labelname, self.labelid)
    def __str__(self) -> str:
        return f"(xi: {self.xi:^ 3} | yi: {self.yi:^ 3}), (x: {self.x:^.3f} | y: {self.y:^.3f}), (w: {self.w:^.3f} | h: {self.h:^.3f}), (objname: {self.labelname} | objid: {self.labelid:^ 5})"
    
class _BaseBoxes:
    def __init__(self) -> None:
        self.GridBox = list()
        self.labelnamelenght = 0

    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self) -> Any:
        if self.idx == len(self.Box):
            raise StopIteration()
        ret = self.GridBox[self.idx]
        self.idx += 1
        return ret
    
    def __str__(self) -> str:
        for bbox in self.GridBox: 
            if self.labelnamelenght < len(bbox.labelname): self.labelnamelenght = len(bbox.labelname)
        output = ""
        for idx, b in enumerate(self.GridBox):
            b = str(b).replace(b.labelname, b.labelname.center(self.labelnamelenght))
            output += f"({idx:^ 3}){b}\n"
        return output
    
    def __len__(self) -> int:
        return len(self.GridBox)
    
    def __iadd__(self, bbox: Union[YoloXYBox, YoloXYCenterIndexBox]) -> None:
        self.GridBox += [bbox]
        return None
    
    def __setitem__(self, idx: int, bbox: Union[YoloXYBox, YoloXYCenterIndexBox]) -> YoloStruct:
        self.GridBox[idx] = bbox
        return self
    
    def __getitem__(self, idx: int) -> Union[YoloXYBox, YoloXYCenterIndexBox]:
        return self.GridBox[idx]

    def __dellitem__(self, idx: int) -> None:
        del self.GridBox[idx]
        return None

@dataclass(frozen=True)
class YoloGridBox:
    S: int
    B: int
    C: int
    def __call__(self) -> Tuple[int, int, int]:
        return (self.S, self.B, self.C)

def PixelRepair(pixel):
    if (pixel - int(pixel)) < 0.5: return int(pixel)
    else: return int(pixel) + 1

class YoloStruct(_BaseBoxes):
    def __init__(self, yolobox: YoloGridBox, bboxes: BoundingBoxes) -> None:
        super(YoloStruct, self).__init__()
        self.yolobox = yolobox
        self.bboxes = bboxes
    
    def Encoder(self) -> YoloStruct:
        S = self.yolobox.S
        if len(self.GridBox) == 0:
            self.bboxes.ToCenter()
            for box in self.bboxes():
                if not isinstance(box, BoundingBoxCenter):
                    raise Exception("not BoundingBoxCenter")
                cx, cy, w, h, name, id = box()
                cxi = int(cx * S)
                cxn = (cx - (cxi / S)) * S
                cyi = int(cy * S)
                cyn = (cy - (cyi / S)) * S
                self.GridBox.append(YoloXYCenterIndexBox(cxi, cyi, cxn, cyn, w, h, name, id))
        else:
            for idx, box in enumerate(self.bboxes()):
                if not isinstance(box, BoundingBoxCenter):
                    raise Exception("not BoundingBoxCenter")
                cx, cy, w, h, name, id = box()
                cxi = int(cx * S)
                cxn = (cx - (cxi / S)) * S
                cyi = int(cy * S)
                cyn = (cy - (cyi / S)) * S
                self.GridBox[idx] = YoloXYCenterIndexBox(cxi, cyi, cxn, cyn, w, h, name, id)
        return self

    def Decoder(self) -> YoloStruct:
        S = self.yolobox.S
        if len(self.GridBox) == 0:
            raise Exception("エンコードされているGridBoxがありません")
        for idx, box in enumerate(self.GridBox):
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            xn = (cxn + cxi) / S
            yn = (cyn + cyi) / S
            xmin = (xn - 0.5 * w) 
            ymin = (yn - 0.5 * h)
            xmax = (xn + 0.5 * w)
            ymax = (yn + 0.5 * h)
            self.GridBox[idx] = YoloXYBox(xmin, ymin, xmax, ymax, name, id)
        return self

    def CreateGridBox(self) -> torch.Tensor:
        S, B, C = self.yolobox() 
        N = B * 5 + C
        Target = torch.zeros(S, S, N)
        for box in self.GridBox:
            (cxi, cyi, cxn, cyn, w, h, name, id) = box()
            Target[cxi, cyi, [0, 1, 2, 3, 5, 6, 7, 8]] = torch.Tensor([cxn, cyn ,w, h, cxn, cyn ,w, h])
            Target[cxi, cyi, [4, 9]] = torch.Tensor([1., 1.])
            Target[cxi, cyi, B * 5 + id] = torch.Tensor([1.])
        return Target

    def GridBoxDetech(self, PredictBox: torch.Tensor, probThreshold: float, Bpatter: bool=False):
        assert PredictBox.dim() < 4 # 0 ~ 3 Clear!
        self.PredictBox = PredictBox
        S, B, C = self.yolobox()
        X = torch.arange(7).unsqueeze(-1).expand(S, S)
        Y = torch.arange(7).unsqueeze(-1).expand(S, S).transpose(1, 0)
        Class = PredictBox[..., 10:].reshape(S, S, C)
        Conf = PredictBox[..., [4, 9]].reshape(S, S, B)
        BBoxes = PredictBox[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape(S, S, B, 4)
        ClassScore, ClassIndex = Class.max(-1)
        maskProb = (Conf * ClassScore.unsqueeze(-1).expand_as(Conf)) > probThreshold
        X = X.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        Y = Y.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        XY = torch.concat((X, Y), dim=-1)
        XYMINMAX = BBoxes[maskProb]
        Conf = Conf[maskProb].unsqueeze(-1)
        ClassIndex = ClassIndex.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        ClassScore = ClassScore.unsqueeze(-1).expand_as(maskProb)[maskProb].unsqueeze(-1)
        bbox = torch.concat((XY, XYMINMAX, Conf, ClassScore, ClassIndex), dim=-1)
        if Bpatter: return bbox
        Result = []
        for b in range(0, bbox.size(0), B):
            box1, box2 = bbox[b], bbox[b+1]
            if torch.sum((box1 == box2).long()) == 9: 
                Result.append(box1)
            else:
                Result.append(box1)
                Result.append(box2)
        return torch.vstack(Result)

    def IoU(self, BBoxP: torch.Tensor, BBoxT: torch.Tensor) -> torch.Tensor:

        if (BBoxP.dim() == 1): BBoxP = BBoxP.unsqueeze(0)
        if (BBoxT.dim() == 1): BBoxT = BBoxT.unsqueeze(0)

        N = BBoxP.size(0)
        M = BBoxT.size(0)

        XYMIN = torch.max(
            BBoxP[..., :2].unsqueeze(1).expand(N, M, 2),
            BBoxT[..., :2].unsqueeze(0).expand(N, M, 2),
        )
        XYMAX = torch.min(
            BBoxP[..., 2:].unsqueeze(1).expand(N, M, 2),
            BBoxT[..., 2:].unsqueeze(0).expand(N, M, 2),
        )

        WH = torch.clamp(XYMAX - XYMIN + 1, min=0)
        Intersection = WH[..., 0] * WH[..., 1]

        Area1 = (BBoxP[..., 2] - BBoxP[..., 0] + 1) * (BBoxP[..., 3] - BBoxP[..., 1] + 1)
        Area2 = (BBoxT[..., 2] - BBoxT[..., 0] + 1) * (BBoxT[..., 3] - BBoxT[..., 1] + 1)
        Area1 = Area1.unsqueeze(1).expand_as(Intersection)
        Area2 = Area2.unsqueeze(0).expand_as(Intersection)

        Union = Area1 + Area2 - Intersection

        return Intersection / Union

    def NMS(self, BBox: torch.Tensor, Scores: torch.Tensor, threshold: float=.5, top_k: int=200):
        """
        Non-Maximum Suppressionを行う関数。
        boxesのうち被り過ぎ(overlap以上)のBBoxを削除する。

        Parameters
        ----------
        boxes : [確信度閾値(0.01)を超えたBBox数,4]
            BBox情報。
        scores :[確信度閾値(0.01)を超えたBBox数]
            confの情報

        Returns
        -------
        keep : リスト
            confの降順にnmsを通過したindexが格納
        count : int
            nmsを通過したBBoxの数
        """

        # returnのひな形を作成
        count = 0
        keep = Scores.new(Scores.size(0)).zero_().long()
        # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

        # 各BBoxの面積areaを計算
        x1 = BBox[:, 0]
        y1 = BBox[:, 1]
        x2 = BBox[:, 2]
        y2 = BBox[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)

        # boxesをコピーする。後で、BBoxの被り度合いIOUの計算に使用する際のひな形として用意
        tmp_x1 = BBox.new()
        tmp_y1 = BBox.new()
        tmp_x2 = BBox.new()
        tmp_y2 = BBox.new()
        tmp_w =  BBox.new()
        tmp_h =  BBox.new()

        # socreを昇順に並び変える
        v, idx = Scores.sort(0)

        # 上位top_k個（200個）のBBoxのindexを取り出す（200個存在しない場合もある）
        idx = idx[-top_k:]

        # idxの要素数が0でない限りループする
        while idx.numel() > 0:
            i = idx[-1]  # 現在のconf最大のindexをiに

            # keepの現在の最後にconf最大のindexを格納する
            # このindexのBBoxと被りが大きいBBoxをこれから消去する
            keep[count] = i
            count += 1

            # 最後のBBoxになった場合は、ループを抜ける
            if idx.size(0) == 1:
                break

            # 現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
            idx = idx[:-1]

            # -------------------
            # これからkeepに格納したBBoxと被りの大きいBBoxを抽出して除去する
            # -------------------
            # ひとつ減らしたidxまでのBBoxを、outに指定した変数として作成する
            tmp_x1 = torch.index_select(x1, 0, idx)
            tmp_y1 = torch.index_select(y1, 0, idx)
            tmp_x2 = torch.index_select(x2, 0, idx)
            tmp_y2 = torch.index_select(y2, 0, idx)

            # すべてのBBoxに対して、現在のBBox=indexがiと被っている値までに設定(clamp)
            tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
            tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
            tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
            tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

            # wとhのテンソルサイズをindexを1つ減らしたものにする
            tmp_w.resize_as_(tmp_x2)
            tmp_h.resize_as_(tmp_y2)

            # clampした状態でのBBoxの幅と高さを求める
            tmp_w = tmp_x2 - tmp_x1
            tmp_h = tmp_y2 - tmp_y1

            # 幅や高さが負になっているものは0にする
            tmp_w = torch.clamp(tmp_w, min=0.0)
            tmp_h = torch.clamp(tmp_h, min=0.0)

            # clampされた状態での面積を求める
            inter = tmp_w*tmp_h

            # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
            rem_areas = torch.index_select(area, 0, idx)  # 各BBoxの元の面積
            union = (rem_areas - inter) + area[i]  # 2つのエリアの和（OR）の面積
            IoU = inter/union

            # IoUがoverlapより小さいidxのみを残す
            idx = idx[IoU.le(threshold)]  # leはLess than or Equal toの処理をする演算です
            # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対してBBoxを囲んでいるため消去

        # whileのループが抜けたら終了

        return keep, count

    def AP(self, Scores: torch.Tensor, Correct: torch.Tensor):
        
        if torch.sum(Correct) == 0:
            return torch.sum(Correct), Correct, Correct

        ScoresSort, IndexSort = torch.sort(Scores, descending=True) # 降順
        Scores = Scores[IndexSort]
        Correct = Correct[IndexSort]

        Precision = torch.cumsum(Correct, dim=-1) / (torch.arange(Correct.size(0)) + 1.) * Correct
        # Recall = torch.cumsum(Correct, dim=-1) / torch.sum(Correct)
        
        # PrecisionFlip = Precision.flip(dims=(0,))
        # PrecisionFlip = torch.cummax(PrecisionFlip, dim=0)[0].flip(dims=(0,))
        
        AP = torch.sum(Precision) / torch.sum(Correct)

        # return AP, Precision, Recall
        return AP

    def MeanAP(self, Predict: torch.Tensor, Target: torch.Tensor, ClassesNum: int, threshold: float=.5):
        Target = self.GridBoxDetech(Target, 0.0, Bpatter=False)
        TBBox = Target[..., [2, 3, 4, 5]].reshape(-1, 4)
        
        Predict = Predict[Target[..., 1].long(), Target[..., 0].long()]
        PBBox = Predict[..., [0, 1, 2, 3, 5, 6, 7, 8]].reshape(-1, 4)
        PConf = Predict[..., [4, 9]].reshape(-1, 2)
        PClass = Predict[..., self.yolobox.B * 5:].argmax(dim=-1).reshape(-1, 1)

        IoU = self.IoU(PBBox, TBBox)
        IoUScores, IoUIndex = torch.max(IoU, dim=-1)
        Correct = (IoUScores >= threshold).float().reshape(-1, 2)
        PScores = torch.concat((PConf, Correct, PClass), dim=-1)
        MAP = 0.
        for idx in range(ClassesNum):
            Classes = PScores[(PScores[..., -1] == idx).unsqueeze(-1).expand_as(PScores)]
            if Classes.size(0) == 0: continue
            Scores = Classes[..., 0].reshape(-1)
            Correct = Classes[..., 1].reshape(-1)
            MAP += self.AP(Scores, Correct)
        MAP /= ClassesNum

        return MAP

if __name__ == "__main__":
    from sklearn.metrics import average_precision_score
    GridBox = YoloGridBox(7, 2, 3)
    BBoxes = BoundingBoxes(400, 400)
    BBoxes += BoundingBox(100, 100, 250, 250, "banana", 0)
    BBoxes += BoundingBox(50, 50, 100, 100, "banana", 1)
    BBoxes += BoundingBox(245, 245, 360, 360, "banana", 2)
    BBoxes.Normalize()
    
    yolostruct = YoloStruct(GridBox, BBoxes)
    yolostruct.Encoder()
    # Target = yolostruct.GridBoxDetech(yolostruct.CreateGridBox(), 0.0, Bpatter=False)
    
    Predict = torch.randn(7, 7, 2 * 5 + 3)
    Predict = torch.sigmoid(Predict)
    MAP = yolostruct.MeanAP(Predict, yolostruct.CreateGridBox(), 3)
    print(MAP)