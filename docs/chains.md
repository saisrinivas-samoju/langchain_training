# Python Basics

Questions:
* What all data structures take position in count? (list, tuple, string does) (dictionary and set doesn't)
* What are *args & **kwargs?


*args -> For any number of positional arguments
```py
def get_prod(*args):
    res = 1
    for arg in args:
        res = res*arg
    return res

get_prod(2, 3, 4, 5)
```
    > 120

**kwargs -> For any number of keywrod arguments

```py
def greet(**kwargs):
    greeting = "Hello"
    if 'name' in kwargs:
        greeting += f", {kwargs['name']}"
    if 'age' in kwargs:
        greeting += f", you are {kwargs['age']} years old"
    if 'location' in kwargs:
        greeting += f" from {kwargs['location']}"
    greeting+="!"
    return greeting

print(greet(name="John"))
print(greet(name="John", age=24))
print(greet(name="John", location='New York'))
print(greet(name="John", age=24, location='New York'))
```
    Hello, John!
    Hello, John, you are 24 years old!
    Hello, John from New York!
    Hello, John, you are 24 years old from New York!

```py
def arg_test(*args):
    return args

arg_test(1, 2, 3)
```
    > (1, 2, 3)


```py
def kwarg_test(**kwargs):
    return kwargs

kwarg_test(a=1, b=3)
```