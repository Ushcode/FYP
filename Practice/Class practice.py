 #!/usr/local/bin/python3

class Snake:
    name = "python" #attribute name of the class
    assoc = [1,2,3]
    
    
    def __init__(self, name):
    self.name = name
    
    
    def change_name(self, new_name): # note that the first argument is self
        self.name = new_name
        
snek = Snake("bro")

print(snek.name)

"""
You can assign the class to a variable. This is called object instantiation. You will then be able to access the attributes that are present inside the class using the dot . operator. For example, in the Snake example, you can access the attribute name of the class Snake.
"""
