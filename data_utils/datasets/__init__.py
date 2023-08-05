from v2xvit.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from v2xvit.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from v2xvit.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from v2xvit.data_utils.datasets.combine_dataset import CombineDataset
from v2xvit.data_utils.datasets.combine_dataset_augmentor import CombineDatasetAugmentor
__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'CombineDataset': CombineDataset,
    'CombineDatasetAugmentor': CombineDatasetAugmentor
}

# the final range for evaluation
GT_RANGE = [-140, -38.4, -3, 140, 38.4, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset

def build_teacher_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = 'EarlyFusionDataset'
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset

def build_distillation_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = 'CombineDatasetAugmentor'
    # error_message = f"{student_dataset_name} is not found. " \
    #                 f"Please add your processor file's name in opencood/" \
    #                 f"data_utils/datasets/init.py"
    # assert student_dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
    #                         'IntermediateFusionDataset'], error_message

    combine_dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )
    return combine_dataset