import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import baseline_utils as b

class BaselineIntegrityTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'filename':f'{s}_{c}.png','patient_id':f'p_{s}',
            'category':c,'split':s} for s in ['train','val','test']
            for c in ['Uninfected','Parasitized']])

    def test_patient_alias_overlap_is_rejected(self):
        d=self.frame(); d.loc[d.split=='val','patient_id']='P-TRAIN'
        with self.assertRaisesRegex(ValueError,'Patient overlap'): b.validate_manifest(d)

    def test_unknown_labels_are_rejected(self):
        d=self.frame(); d.loc[0,'category']='infected?'
        with self.assertRaisesRegex(ValueError,'category'): b.validate_manifest(d)

    def test_positive_class_and_confusion_orientation(self):
        d=b.validate_manifest(self.frame())
        self.assertTrue((d.loc[d.category=='Parasitized','label']==1).all())
        m=b.classification_metrics([0,1,1,0],[0,0,1,1],[-.7,-.2,.8,.4])
        self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertAlmostEqual(m['f1'],.5)

    def test_shared_scaled_images_are_not_scaled_twice(self):
        image=np.full((4,4,3),.5,dtype=np.float32)
        with patch.object(b.shared,'load_and_pad',return_value=image), patch.object(b.shared,'scale_pixels') as scale:
            fn,backend=b.choose_preprocessor('unused',4)
            np.testing.assert_array_equal(fn('unused'),image)
            scale.assert_not_called()
            self.assertEqual(backend,'team_shared_scaled_loader')

    def test_shared_raw_pixels_use_shared_scaler_once(self):
        raw=np.full((4,4,3),128,dtype=np.uint8)
        with patch.object(b.shared,'load_and_pad',return_value=raw), patch.object(b.shared,'scale_pixels',side_effect=lambda x:x.astype(np.float32)/255) as scale:
            fn,backend=b.choose_preprocessor('unused',4)
            np.testing.assert_allclose(fn('unused'),128/255)
            self.assertEqual(scale.call_count,2)  # probe plus one actual image
            self.assertEqual(backend,'team_shared_load_and_scale')


    def test_shared_loader_preserves_frozen_baseline_pixels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PIL import Image
        image = (np.arange(17*23*3).reshape(17,23,3) % 256).astype(np.uint8)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'cell.png'
            Image.fromarray(image).save(path)
            for size in (8,32,128):
                shared = b.shared.load_and_pad(str(path),size)
                frozen = b.legacy_load_and_pad(path,size)
                np.testing.assert_array_equal(shared,frozen)

if __name__=='__main__': unittest.main()
