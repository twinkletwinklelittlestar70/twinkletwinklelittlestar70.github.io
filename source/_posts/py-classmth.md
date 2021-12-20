---
title: Fix error when Python class method called by a multi thread task
date: 2021-10-18 21:17:34
layout: python multi-thread class-method
---

Here is the story. I am developing a python flask project today. And I need to provide an interface for the front end to run a model to recognize a set of facial images. 

For a group of ten images, I first developed a single-threaded version to recognize, and the program took 2.4s in total. So I think, will there be optimization for multi-threading? I decided to use two threads and wrote the following code very quickly.

```python
class TaskThread(threading.Thread):
    def __init__(self, func, args=()):
        super(TaskThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print('[ERROR] in TaskThread ', e)
            return None

class RecogModel():
    # ....
    def __init__(self):
        # Load the model
        self.model = load_model()
        pass
    
    def predictImage (self, filepath):
        # ... model predict
        return result
    
    def predict (self, id, is_multi_thread = False):
        # ... get filepath
        
        work_task = TaskThread(self.predictImage, (filepath))
        work_task.start()
        
        # ....

```

However I got error like this.

```
TypeError: method() takes 1 positional argument but 2 were given
```

As we all know, the python class method feature is that when you call the internal method, you don't need to manually pass the this pointer (self). When python is executed, it will add the first parameter `this` to the internal method by default. The code should be like this.


```python
class RecogModel():
    def __init__(self):
        # Load the model
        self.model = load_model()
        pass
    
    def predictImage (self, filepath):
        # ... do model prediction

        return result
    
    def predict (self, id, is_multi_thread = False):
        # ... get filepath
        
        result = self.predictImage(filepath) # no need to pass self
        
        # ... do something

```

We can see that even though we only passed one parameter when calling the `self.predictImage` method. But in actual operation, python will add the self pointer to the `predictImage` method. This feature helps us reduce repetitive writing of "self" code, which is very convenient in most cases.

But when I delayed the class internal method to be called inside a thread class, the problem appeared. The code we actually call is:

```python
self.result = self.func(*self.args)
```

Due to our manual call, the python class cannot correctly bind the self parameter to our call.

The first solution I thought of was to curry the function that needs to be called and turn it into a static method that does not need to bind self. Code show as below.

```python
class RecogModel():
    def __init__(self):
        # Load the model
        self.model = load_model()
        pass
    
    @staticmethod
    def predictImage (model, filepath):
        # ... do model prediction

        return result
    
    def predict (self, id, is_multi_thread = False):
        # ... get filepath
        
        result = self.predictImage(self.model, filepath)
        
        # ... do something
```

This way can work, but I think it is not good  enough. A static method should be a tool method that is related to the class, but can be called without an instance of the class. Obviously this is not the case in our example. In our example, method `predictImage` needs a internal variable(the model). So method `predictImage` should be a member function.

In the end, my solution was very simple, using `__func__` in the python function object to get the original function and passing self manually. 

My final code is as follows.

```python

work_task = ThreadClass.TaskThread(self.predictImageList.__func__, (self, file_list))

```


Finally, let's take a look at the complete code.

```python
class TaskThread(threading.Thread):
    def __init__(self, func, args=()):
        super(TaskThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            print('[ERROR] in TaskThread ', e)
            return None

class RecogModel():
    # ....
    def __init__(self):
        # Load the model
        self.model = load_model()
        pass
    
    def predictImage (self, filepath):
        # ... model predict
        return result
    
    def predict (self, id, is_multi_thread = False):
        # ... get filepath
        
        work_task = ThreadClass.TaskThread(self.predictImageList.__func__, (self, file_list))
        work_task.start()
        
        # ....

```

By the way, multithreading did not help me in time cost. The 2-thread version takes about 2.3 seconds, and the 10-thread version takes more than 4 seconds.

I guess life is always unsatisfactory, we can only accept it. Still hope this can help you :)