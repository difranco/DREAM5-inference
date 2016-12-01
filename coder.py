from collections import defaultdict
from sys import intern

class Coder:
    """
    Creates sequential ids for objects and keeps track of them
    useful e.g. for filling out a matrix with info related to objects
    and translating back and forth between matrix indices and the objects
    """
    def next_id(self):
        self.current_id += 1
        return self.current_id
    
    def total_seen(self):
        return self.current_id + 1
    
    def __init__(self):
        self.current_id = -1
        self.obj_to_int = defaultdict(self.next_id)
        self.int_to_obj = dict()

    def encode(self, obj):
        if isinstance(obj, str):
            obj = intern(obj)
        code = self.obj_to_int[obj]
        self.int_to_obj[code] = obj
        return code
    
    def decode(self, i):
        return self.int_to_obj[i]

    def get_code(self, obj):
        """
        Gets the code for an object but won't extend the code
        if the object isn't already present
        """
        if isinstance(obj, str):
            obj = intern(obj)
        if obj in self.obj_to_int: return self.obj_to_int[obj]
        else: return None

