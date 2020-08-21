import os.path

import Orange
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.spectroscopy.widgets.owptirfile import OWPTIRFile
from orangecontrib.spectroscopy import get_sample_datasets_dir

PHOTOTHERMAL_HYPER = "photothermal/Hyper_on_H_Treated_Sample.ptir"
PHOTOTHERMAL_ARRAY = "photothermal/Nodax_Spectral_Array.ptir"


class TestOWPTIRFile(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPTIRFile)

    def test_load_hyperspectral(self):
        path = os.path.join(get_sample_datasets_dir(), PHOTOTHERMAL_HYPER)
        self.widget.add_path(path)
        self.widget.source = self.widget.LOCAL_FILE
        self.widget.load_data()
        self.wait_until_stop_blocking()
        self.assertNotEqual(self.get_output("Data"), None)

    def test_load_array(self):
        path = os.path.join(get_sample_datasets_dir(), PHOTOTHERMAL_ARRAY)
        self.widget.add_path(path)
        self.widget.source = self.widget.LOCAL_FILE
        self.widget.load_data()
        self.wait_until_stop_blocking()
        self.assertNotEqual(self.get_output("Data"), None)
