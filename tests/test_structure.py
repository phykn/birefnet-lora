import subprocess
import sys
from pathlib import Path

from src.build.split import save as legacy_save_splits
from src.data.dataset import MaskDataset
from src.data.split import save as save_splits
from src.predict.inference import predict_logits
from src.predict.run import predict_logits as legacy_predict_logits
from src.prepare.load import MaskDataset as LegacyMaskDataset
from src.train.objective import TrainLoss
from src.train.loss import TrainLoss as LegacyTrainLoss
from src.train.run import Trainer as LegacyTrainer
from src.train.trainer import Trainer


def test_compatibility_exports_point_to_canonical_owners():
    assert LegacyMaskDataset is MaskDataset
    assert legacy_save_splits is save_splits
    assert legacy_predict_logits is predict_logits
    assert LegacyTrainLoss is TrainLoss
    assert LegacyTrainer is Trainer


def test_model_output_import_stays_light_and_birefnet_export_is_compatible():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from src.model.output import Output\n"
        "assert Output.__name__ == 'Output'\n"
        "assert 'src.model.net' not in sys.modules\n"
        "assert 'src.model.swin' not in sys.modules\n"
        "assert 'torchvision' not in sys.modules\n"
        "import src.model as model\n"
        "assert 'Output' in model.__all__\n"
        "assert 'BiRefNet' in model.__all__\n"
        "from src.model import BiRefNet\n"
        "assert BiRefNet.__name__ == 'BiRefNet'\n"
        "assert 'src.model.net' in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
