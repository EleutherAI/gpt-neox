import tensorflow as tf
import pandas as pd
import numpy as np
class TFrecordCreator:
    """Creates TFRecord Dataset to store the memorization metric
    """
    def __init__(self,path):
        self.path = path
        self.writer = tf.io.TFRecordWriter(path)
    
    def _int64_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _float_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def _serialize_example(self,result,index):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature = {
            'result':self._float_feature(result),
            'index':self._int64_feature(index)
        }
        
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def write(self,result,index):
        value = self._serialize_example(result,index)
        self.writer.write(value)
    
    def close(self):
        """Closes the tfrecord stream
        
        For some reason, using __del__ doesn't work
        """
        self.writer.close()




class TFRecordLoader:
    """Loads the data created by  TFrecordCreator

    """
    def __init__(self,path):
        self.path = path
        self.reader = tf.data.TFRecordDataset([path])
        self.reader = self.reader.map(self._parse_fn)
        self.reader = self.reader.apply(tf.data.experimental.ignore_errors())
    
    def _parse_fn(self,example_proto):

        feature_description = {
            'result':tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'index':tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }
        res =  tf.io.parse_single_example(example_proto, feature_description)
        return res['result'],res['index'] 
    def __iter__(self):
        return iter(self.reader)


class DataFrameCreator:
    def __init__(self,path):
        self.path = path
        self.fp = open(path,'w')
        self.fp.write("index,nll_loss,accuracy\n")
        self.fp.flush()
    
    def write(self, index, nll_loss, accuracy):
        self.fp.write(f'{index},{nll_loss},{accuracy}\n')
        self.fp.flush()
    
    def close(self):
        self.fp.close()

class DataFrameLoader:
    def __init__(self,path):
        csv = pd.read_csv(path)
        self.csv = np.array(csv)
    
    def __getitem__(self,idx):
        return self.csv[idx,:]
    
    def __len__(self):
        return len(self.csv)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

if __name__ == '__main__':
    ds = DataFrameCreator('temp.csv')
    for i in range(10):
        ds.write(i,i+0.5,i)
    ds.close()
    del ds
    ds = DataFrameLoader('/home/mchorse/gpt-neox/memorization_results_dense_small_checkpoints.csv')
    for i in ds:
        print(i)