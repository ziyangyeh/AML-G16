import monai.metrics
from monai.utils import MetricReduction

from .build_losses_metrics import METRICS

@METRICS.register_module()
class mDice(monai.metrics.DiceMetric):
    def __init__(self, include_background: bool = True, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False, ignore_empty: bool = True, num_classes: int | None = None) -> None:
        super(mDice, self).__init__(include_background, reduction, get_not_nans, ignore_empty, num_classes)

@METRICS.register_module()
class mIoU(monai.metrics.MeanIoU):
    def __init__(self, include_background: bool = True, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False, ignore_empty: bool = True) -> None:
        super(mIoU, self).__init__(include_background, reduction, get_not_nans, ignore_empty)
