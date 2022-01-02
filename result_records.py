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
        self.indicies = []
        self.results = []
    
    def write(self,result,index):
        self.results.append(result)
        self.indicies.append(index)
    
    def close(self):
        csv = pd.DataFrame()
        csv['index'] = self.indicies
        csv['result'] = self.results
        csv.to_csv(self.path,index=False)

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
        ds.write(i,i)
    ds.close()
    del ds
    ds = DataFrameLoader('/home/mchorse/gpt-neox/memorization_results_neox_dense_large_v2.csv')
    for i in ds:
        print(i)