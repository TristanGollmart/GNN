from tensorflow import keras
import os
zip_file = keras.utils.get_file(fname='cora.tgz',
                                origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
                                extract=True,
                                )
data_dir = os.path.join(os.path.dirname(zip_file), "cora")
#test