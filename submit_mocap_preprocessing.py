import h5py
import pickle
import numpy as np
import os
from typing import Text, Dict
import subprocess

# folders = [
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_21_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_22_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_23_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_24_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_25_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_26_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_27_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_28_3",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_29_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_30_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2020_12_31_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_01_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_02_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_04_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_05_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_06_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_07_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_09_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_09_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_10_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_10_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_11_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_25_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_29_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_01_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_07_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/art/2021_02_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_21_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_22_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_25_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_29_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_06_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_02_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_05_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_07_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_10_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_13_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_14_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_29_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_31_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_02_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_04_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_05_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_07_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_09_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_10_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_13_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_14_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_23_2",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_25_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_31_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_04_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_14_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_25_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_04_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_05_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_09_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_10_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_13_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_04_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_10_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_18_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_19_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_22_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_23_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_25_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_28_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_31_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_02_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_09_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_13_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_05_30_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_05_31_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_01_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_02_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_03_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_06_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_07_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_08_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_09_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_11_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_12_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_13_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_14_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_15_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_16_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_17_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_20_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_21_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_24_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_26_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_27_1",
# "/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_29_1",
# ]

folders = [
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_07_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_08_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_10_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_11_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_13_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_14_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/bud/2021_07_19_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_28_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_29_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_30_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_07_31_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_01_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_02_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_03_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_04_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_05_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_06_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_07_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_08_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_09_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_10_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_11_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_12_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_13_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_14_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_19_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_20_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_23_2",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_25_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_27_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_30_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_08_31_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_03_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_04_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_14_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/coltrane/2021_09_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_19_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_20_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_22_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_23_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_24_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_25_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_26_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_27_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_02_28_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_01_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_03_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_04_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_05_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_06_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_08_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_09_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_10_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_11_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_12_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_13_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_19_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_22_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/duke/2022_03_23_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_04_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_10_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_21_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_22_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/espie/2022_03_23_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_18_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_19_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_20_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_21_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_22_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_23_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_24_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_25_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_26_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_27_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_28_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_30_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_05_31_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_01_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_02_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_03_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_06_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_08_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_09_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_12_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_13_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/freddie/2022_06_20_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_05_30_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_05_31_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_01_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_02_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_03_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_06_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_07_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_08_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_09_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_11_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_12_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_13_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_14_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_15_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_16_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_17_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_20_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_21_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_24_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_26_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_27_1",
"/n/holylabs/LABS/olveczky_lab/holylfs02/Everyone/dannce_rig/dannce_ephys/gerry/2022_06_29_1",
]
class NpmpPreprocessingDispatcher:

    """Dispatch preprocessing jobs to a hpc.

    Attributes:
        clip_end (int): Last step in clip.
        clip_length (int): Number of steps in clip.
        end_steps (List): End steps for each batch.
        save_folder (Text): Folder in which to save data.
        stac_path (Text): Path to stac file containing reference.
        start_steps (List): Start steps for each batch.
    """

    def __init__(
        self,
        stac_path: Text,
        save_folder: Text,
        clip_length: int = 2500,
    ):
        """Initialize NpmpPreprocessingDispatcher

        Args:
            stac_path (Text): Path to stac file containing reference.
            save_folder (Text): Folder in which to save data.
            clip_length (int, optional): Number of steps in clip.
        """
        self.stac_path = stac_path
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        self.clip_length = clip_length
        self.clip_end = self.get_clip_end()

        self.start_steps = np.arange(0, self.clip_end, self.clip_length)
        self.end_steps = self.start_steps + self.clip_length
        self.end_steps[-1] = self.clip_end
        batch_args = []
        self.batch_args_file = os.path.join(os.path.dirname(self.save_folder), "_batch_preprocessing_args.p")
        for start, end in zip(self.start_steps, self.end_steps):
            batch_args.append(
                {
                    "stac_path": self.stac_path,
                    "save_file": os.path.join(self.save_folder, "%d.hdf5" % (start)),
                    "start_step": start,
                }
            )
        self.save_batch_args(batch_args, self.batch_args_file)

    def get_clip_end(self) -> int:
        """Get the last frame in the clip from the reference.

        Returns:
            int: End frame in clip.
        """
        with h5py.File(self.stac_path, "r") as file:
            n_samples = file["qpos"][:].shape[0]
        # data = loadmat(self.stac_path)
        # n_samples = data["qpos"][:].shape[0]
        # with open(self.stac_path, "rb") as f:
        #     in_dict = pickle.load(f)
        #     n_samples = in_dict["qpos"].shape[0]
        return n_samples

    def dispatch(self):
        """Submit the job."""
        script = f"""#!/bin/bash
#SBATCH --job-name=preprocessNPMP
#SBATCH --mem=5000
#SBATCH -t 0-10:00
# # SBATCH --array=0-{len(self.start_steps) - 1}
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared,serial_requeue,cox
#SBATCH --exclude=holy2c18111 #seasmicro25 was removed
#SBATCH --constraint="intel&avx2"
source ~/.bashrc
mj_sing python -c "import mocap_preprocess as mcp; mcp.npmp_embed_preprocessing_single_batch('{self.batch_args_file}')"
"""
        job_id = subprocess.check_output(
            "sbatch", input=script, universal_newlines=True
        )
        job_id = job_id.strip().split()[-1]
        
        merge_script = f"""#!/bin/bash
#SBATCH --job-name=npmp_merge
#SBATCH --mem=40000
#SBATCH -t 0-01:00
#SBATCH --dependency=afterok:{job_id}
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p olveczky,shared,cox,serial_requeue
#SBATCH --constraint="intel&avx2"
set -e

# Load the environment setup functions
source ~/.bashrc

mj_sing python -c "import mocap_preprocess as mcp; mcp.merge_preprocessed_files('{self.save_folder}')"
"""
        job_id = subprocess.check_output(
            ["sbatch", "--wait"], input=merge_script, universal_newlines=True
        )

    def save_batch_args(self, batch_args: Dict, batch_args_path: str):
        """Save the batch arguments to file.

        Args:
            batch_args (Dict): Dictionary of arguments to batches.
        """
        with open(batch_args_path, "wb") as f:
            pickle.dump(batch_args, f)


def submit():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = folders[task_id]
    stac_file = os.path.join(folder, "stac", "total.hdf5")
    save_folder = os.path.join(folder, "npmp_preprocessing")
    dispatcher = NpmpPreprocessingDispatcher(stac_file, save_folder)
    dispatcher.dispatch()

def submit_all():
    script = f"""#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH -p olveczky,shared,cox
#SBATCH -n 1 # Number of cores/tasks
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=1000
#SBATCH -t 0-05:00:00
#SBATCH --array=0-{len(folders)-1}%50
source ~/.bashrc
source /n/home02/daldarondo/miniconda3/bin/activate error_analysis_env
python -c "import submit_mocap_preprocessing as smp; smp.submit()"
"""
    job_id = subprocess.check_output(
        "sbatch", input=script, universal_newlines=True
    )