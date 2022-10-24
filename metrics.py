import torch


class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes = exclude_classes

    def update(self, a, b):
        a = a.cpu()
        b = b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds = inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global = acc_global.item() * 100
        acc = (acc * 100).tolist()
        iu = (iu * 100).tolist()
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global = round(acc_global, 2)
        IOU = [round(i, 2) for i in iu]
        mIOU = sum(iu) / len(iu)
        mIOU = round(mIOU, 2)
        reduced_iu = [
            iu[i] for i in range(self.num_classes) if i not in self.exclude_classes
        ]
        mIOU_reduced = sum(reduced_iu) / len(reduced_iu)
        mIOU_reduced = round(mIOU_reduced, 2)

        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"
