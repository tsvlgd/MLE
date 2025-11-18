# @staticmethod

class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b
    
    @staticmethod
    def multiply(a, b):
        return a * b

# Usage
result = MathUtils.add(5, 3)  # No instance or class reference needed
print (result)  # Output: 8
result_multiply = MathUtils.multiply(5, 3)  # No instance or class reference needed
print(result_multiply)  # Output: 15




# @classmethod

class MyClass:
    """
    What it is:
    A method that takes cls (the class itself) as its first parameter instead of self.
    refers to the class, not the instance.
    They cannot directly access or modify instance attributes (variables defined in __init__ or other instance methods) but can access class attributes.
    
    When to Use @classmethod?

    Factory Methods: Create instances in a non-standard way.
    Class-Level Operations: Modify or access class attributes.
    Alternative Constructors: Provide flexible ways to create objects.
    """
    class_variable = 0

    def __init__(self, value):
        self.instance_variable = value
        MyClass.class_variable += 1

    @classmethod
    def increment_class_variable(cls):
        cls.class_variable += 1
        print(f"Class variable incremented to: {cls.class_variable}")

    @classmethod
    def create_from_string(cls, data_string):
        # An example of a factory method
        value = int(data_string.split('-')[1])
        return cls(value)

# Calling an instance method (implicitly calls __init__)
obj1 = MyClass(10)
print(f"Instance 1 class_variable: {MyClass.class_variable}")

# Calling a class method
MyClass.increment_class_variable()

# Creating an instance using a class method (factory method)
obj2 = MyClass.create_from_string("data-20")
print(f"Instance 2 value: {obj2.instance_variable}")
print(f"Instance 2 class_variable: {MyClass.class_variable}")





class Counter:
    """
    The class method get_total_instances accesses the class attribute count to return the total number of instances created.
    """
    count = 0  # Class attribute

    def __init__(self):
        Counter.count += 1  # Modify class attribute

    @classmethod
    def get_total_instances(cls):
        return cls.count  # Access class attribute

# Usage
a = Counter()
b = Counter()
print(Counter.get_total_instances())  # Output: 2
