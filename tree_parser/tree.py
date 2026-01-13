# import os
# from tree_parser.node import Node

# OUTPUT_DIR = ".outputs/"

# class Tree:
#     def __init__(self, file):
#         self.rootNode = Node('0', "root", os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(file))[0]))
#         self.file = file
    
import os
from tree_parser.node import Node

class Tree:
    def __init__(self, file, user_param: str, output_dir: str = None):
        output_dir = os.path.join(output_dir if output_dir else os.path.expanduser("~"), user_param)
        os.makedirs(output_dir, exist_ok=True)

        self.rootNode = Node(
            '0', 
            "root", 
            os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0])
        )
        self.file = file
