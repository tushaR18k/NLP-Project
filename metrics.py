import torch
import warnings
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # print(f'res{res}')
        return res


class Metrics(object):
    """
    metric is an abstract class.
    Args:
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, average=True, **kwargs):
        self._average = average
        self.eps = 1e-20
        self.reset()
        self.result = torch.FloatTensor()

    def reset(self):
        """Reset the private values of the class."""
        raise NotImplementedError

    def update(self, output=None, target=None):
        """
        Main calculation of the metric which updated the private values respectively.
        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        raise NotImplementedError

    def calculate_result(self):
        """calculate the final values when the epoch/batch loop
        is finished.
        """
        raise NotImplementedError

    @property
    def avg(self):
        warnings.warn('`avg` is deprecated, please use `value`.', DeprecationWarning)
        return self.value

    @property
    def value(self):
        """output the metric results (array shape) or averaging
        out over the results to output one single float number.
        Returns:
            result (np.array / float): final metric result
        """
        self.result = torch.FloatTensor(self.calculate_result())
        if self._average and self.result.numel() == self.result.size(0):
            return self.result.mean(0).cpu().numpy().item()
        elif self._average:
            return self.result.mean(0).cpu().numpy()
        else:
            return self.result.cpu().numpy()

    @property
    def standard_dev(self):
        """Return the standard deviation of the metric."""
        result = torch.FloatTensor(self.calculate_result())
        if result.numel() == result.size(0):
            return result.std(0).cpu().numpy().item()
        else:
            return result.std(0).cpu().numpy()

    def __str__(self):
        val = self.value
        std = self.standard_dev
        if isinstance(val, np.ndarray):
            return ", ".join(f"{v:.3f}±{s:.3f}" for v, s in zip(val, std))
        else:
            return f"{val:.3f}±{std:.3f}"


class Precision_class(Metrics):
    """computes the precision for each class over epochs.
    Args:
        num_classes (int): number of classes.
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, num_classes: int, average=True, **kwargs):
        self.n_class = num_classes
        super().__init__(average=average)
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)

    def reset(self):
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._false_positives = torch.zeros([self.n_class], dtype=torch.float32)

    def update(self, output=None, target=None):
        """
        Update tp, fp and support acoording to output and target.
        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        # (batch, 1)
        target = target.view(-1)

        # (batch, nclass)
        indices = torch.argmax(output, dim=1).view(-1)

        output = indices.type_as(target)
        correct = output.eq(target.expand_as(output))
        # print(f'output{output}')
        # print(f'target{target}')

        # Convert from int cuda/cpu to double cpu
        for class_index in target:
            self._positives[class_index] += 1
        for class_index in indices[(correct == 1).nonzero()]:
            self._true_positives[class_index] += 1
        for class_index in indices[(correct == 0).nonzero()]:
            self._false_positives[class_index] += 1

    def calculate_result(self):
        # print(f'true_pos={self._true_positives}')
        # print(f'false_pos={self._false_positives}')
        result = self._true_positives / self._positives
        # precision_div=self._true_positives+self._false_positives
        # result = self._true_positives / precision_div if precision_div != 0 else 0

        # where the class never was shown in targets
        result[result != result] = 0

        return result

    def __str__(self):
        return f'Precision: {torch.mean(self.calculate_result())}'


class Recall_class(Metrics):
    """computes the precision for each class over epochs.
    Args:
        num_classes (int): number of classes.
        average (bool): a way to output one single value for metrics
                        that are calculated in several trials.
    """

    def __init__(self, num_classes: int, average=True, **kwargs):
        self.n_class = num_classes
        super().__init__(average=average)
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._label_imageid = [[] for i in range(self.n_class)]

    def reset(self):
        self._true_positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._positives = torch.zeros([self.n_class], dtype=torch.float32)
        self._false_negatives = torch.zeros([self.n_class], dtype=torch.float32)
        self._label_imageid = [{} for i in range(self.n_class)]

    def update(self, image_id, output=None, target=None):
        """
        Update tp, fp and support acoording to output and target.
        Args:
            output (tensor): predictions of model
            target (tensor): labels
        """
        # (batch, 1)
        target = target.view(-1)

        # (batch, nclass)
        indices = torch.argmax(output, dim=1).view(-1)

        output = indices.type_as(target)
        correct = output.eq(target.expand_as(output))
        # print(f'output{output}')
        # print(f'target{target}')

        # Convert from int cuda/cpu to double cpu
        # for class_index in target:
        for class_index in output:
            self._positives[class_index] += 1
        for idx, class_index in enumerate(indices[(correct == 1).nonzero()]):
            self._true_positives[class_index] += 1
            self._label_imageid[idx].append(image_id[idx])
        for class_index in target[(correct == 0).nonzero()]:
            # false_negatives[class_index] += 1
            self._false_negatives += 1

    def calculate_result(self):
        # print(f'true_pos={self._true_positives}')
        # print(f'false_pos={self._false_positives}')

        result = self._true_positives / self._positives
        total = self._positives.float().sum().item()
        # print(total)
        distb = self._true_positives / total
        # recall_div=self._true_positives+self._false_negatives
        # result = self._true_positives / recall_div if recall_div != 0 else 0

        # where the class never was shown in targets
        result[result != result] = 0

        return result, distb

    def __str__(self):
        result, distb = self.calculate_result()
        # print(distb)
        # print(self._label_imageid)
        return f'Recall: {torch.mean(result)}'
