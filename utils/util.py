from PIL import Image
import os
from pathlib import Path

class auto_util():

    def __init__(self):
        super().__init__()

    def get_project_root(self) -> Path:
        return Path(__file__).parent.parent

    def find_files(self, filename, search_path):
        result = []
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result

    def show_module_details(self):
        os.system('pipdeptree -p autovariate --graph-output png > graph.png')
        img = Image.open('graph.png')
        img.show()
        graph = self.find_files("graph.png", ".")[0]
        os.remove(graph)

    
    

    



            
 


