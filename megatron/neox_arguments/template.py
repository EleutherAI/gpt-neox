from dataclasses import dataclass
import logging 

@dataclass
class NeoXArgsTemplate:

    def defaults(self):
        """
        generator for getting default values.
        """
        for key, field_def in self.__dataclass_fields__.items():
            yield key, field_def.default
    
    def update_value(self, key: str, value):
        """
        updates a property value if the key already exists

        Problem: a previously non-existing property can be added to the class instance without error. 
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            error_message = self.__class__.__name__+".update_value() to be updated property "+str(key)+" does not exist"
            logging.error(error_message)
            raise ValueError(error_message)
    
    def update_values(self, d):
        """
        Updates multiple values in self if the keys already exists
        """
        for k, v in d.items():
            self.update_value(k, v)