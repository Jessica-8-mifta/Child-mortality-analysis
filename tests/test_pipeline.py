import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from child_mortality.data import validate
from child_mortality.models import metrics
from child_mortality.pipeline import run

class PipelineTests(unittest.TestCase):
    def sample(self):
        rng = np.random.default_rng(13)
        n = 45
        return pd.DataFrame({'iso3': ['BGD'] + [f'X{i:02}' for i in range(n-1)], 'year': 2019,
            'u5_mort': rng.uniform(3, 70, n), 'gdp_pc': rng.uniform(500, 50000, n),
            'measles_imm': rng.uniform(50, 99, n), 'water': rng.uniform(40, 100, n),
            'sanit': rng.uniform(20, 100, n)})

    def test_validation(self):
        df = self.sample()
        df.loc[0, 'gdp_pc'] = 0
        df.loc[1, 'water'] = np.nan
        clean, audit, _ = validate(df, 2019)
        self.assertEqual(len(clean), 43)
        self.assertEqual(audit['included'].sum(), 43)
        df.loc[2, 'iso3'] = df.loc[3, 'iso3']
        with self.assertRaises(ValueError):
            validate(df, 2019)

    def test_failed_predictions_never_dropped(self):
        with self.assertRaises(ValueError):
            metrics([1, 2], [1, np.nan])

    def test_end_to_end_and_split_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.sample().to_csv(root / 'input.csv', index=False)
            args = SimpleNamespace(data=str(root / 'input.csv'), year=2019, seed=42,
                                   output=str(root / 'run'), cache=str(root / 'cache'))
            run(args)
            splits = pd.read_csv(root / 'run/splits.csv').set_index('iso3')['split']
            self.assertEqual(splits['BGD'], 'bangladesh_case')
            oof = pd.read_csv(root / 'run/oof_predictions.csv')
            self.assertTrue(set(oof['iso3']).isdisjoint(set(splits[splits != 'train'].index)))
            test = pd.read_csv(root / 'run/test_predictions.csv')
            self.assertEqual(set(test['iso3']), set(splits[splits == 'test'].index))
            self.assertTrue((root / 'run/model.joblib').exists())
