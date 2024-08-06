#!/usr/bin/env python
# coding: utf-8

# Q.1) Ans:
# To perform a matrix multiplication between a (n, k) and (k, m) matrix we need to perform:
# 
# Number of multiplications: n×m×k ,
# Number of additions: n×m×(k−1)
# 

# Q.2) Ans:
# Solve using using both a list of lists and NumPy.Compareing the timing of both solutions.

# In[33]:


#lets consider 2 array
#2*3  and  3*3= 2*3

List1=[[4,6],[7,2]]
List2=[[9,4,2],[7,3,8]]

newarr1=[[0,0,0],[0,0,0]]
for i in range(0,len(newarr1)):
    for j in range(0,len(newarr1[0])):
        for k in range(0,len(List2)):
            newarr1[i][j]+=List1[i][k]*List2[k][j]
for row in newarr1:
    print(row)

get_ipython().run_line_magic('timeit', 'for row in newarr1:row')


# In[34]:


import numpy as np

arr1=np.array([[4,6],[7,2]])
arr2=np.array([[9,4,2],[7,3,8]])
newarr1=np.matmul(arr1,arr2)
#newarr=np.dot(arr1,arr2)
print(newarr1)
get_ipython().run_line_magic('timeit', 'newarr1')


# By running this complete code it seems NumPy is significantly faster due to its optimized operations and efficient memory usage.

# Q.4) Ans:
# 
# f = x^2y + y^3sin(x)
# 
# gradient_f = ((d/dx)i + (d/dy)j)*f
# 
# gradient_x = 2xy + y^3*cos(x)
# 
# gradient_y = x^2 + 3y^2*sin(x)
# 
# The gradient of the function is,
# 
# ∆f = 2xy + y^3cos(x) + x^2 + 3y^2sin(x)

# Q.5) Use JAX to confirm the gradient.

# In[35]:


import jax
import numpy as np

def func(x,y):
    return x**2 *y+y**3 *jax.numpy.sin(x)

x_value=4.0     #taking random value to compare solution
y_value=4.0

#solution using jax

grad_func=jax.grad(func,argnums=(0,1))
gradient=grad_func(x_value,y_value)
print(gradient)

#analytical solution

grad=np.array([2*x_value*y_value+y_value**3 *np.cos(x_value),x_value**2+3*y_value**2 *np.sin(x_value)])
print(grad)


# Q6):Useing sympy

# In[36]:


import sympy
from sympy import diff,sin,cos
from sympy.abc import x,y


func=x**2 *y+y**3 *sin(x)
#expression_x=2*x*y+y**3 *cos(x)
#expression_y=x**2+3 *y**2 *sin(x)

grad_f=(diff(func,x),diff(func,y))
grad_f
x_value=4.0
y_value=4.0

numerical_gradient = [grad_f[0].subs({x: x_value, y: y_value}), grad_f[1].subs({x: x_value, y: y_value})]
numerical_gradient


# Q7): Createing a Python nested dictionary to represent hierarchical information.

# In[37]:


students_records = {
    2022: {
        "Branch 1": {
            1: {
                "Name": "N",
                "Marks": {
                    "Maths": 100,
                    "English": 70
                }
            }
        },
        "Branch 2": {}
    },
    2023: {
        "Branch 1": {},
        "Branch 2": {}
    },
    2024: {
        "Branch 1": {},
        "Branch 2": {}
    },
    2025: {
        "Branch 1": {},
        "Branch 2": {}
    }
}

# Print the dictionary to verify the structure
import pprint
pprint.pprint(students_records)


# Q8):To represent the hierarchical information using Python classes.

# In[38]:


class Marks:
    def __init__(self, **kwargs):
        self.subjects = kwargs

    def __repr__(self):
        return repr(self.subjects)

class Student:
    def __init__(self, roll_number, name, marks):
        self.roll_number = roll_number
        self.name = name
        self.marks = marks

    def __repr__(self):
        return f"Student(Roll Number: {self.roll_number}, Name: {self.name}, Marks: {self.marks})"

class Branch:
    def __init__(self, name):
        self.name = name
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def __repr__(self):
        return f"Branch(Name: {self.name}, Students: {self.students})"

class Year:
    def __init__(self, year):
        self.year = year
        self.branches = []

    def add_branch(self, branch):
        self.branches.append(branch)

    def __repr__(self):
        return f"Year({self.year}, Branches: {self.branches})"

# Creating the database
database = []

# Adding data for 2022
year_2022 = Year(2022)
branch_1_2022 = Branch("Branch 1")
branch_1_2022.add_student(Student(1, "N", Marks(Maths=100, English=70)))
year_2022.add_branch(branch_1_2022)
year_2022.add_branch(Branch("Branch 2"))
database.append(year_2022)

# Adding data for 2023
year_2023 = Year(2023)
year_2023.add_branch(Branch("Branch 1"))
year_2023.add_branch(Branch("Branch 2"))
database.append(year_2023)

# Adding data for 2024
year_2024 = Year(2024)
year_2024.add_branch(Branch("Branch 1"))
year_2024.add_branch(Branch("Branch 2"))
database.append(year_2024)

# Adding data for 2025
year_2025 = Year(2025)
year_2025.add_branch(Branch("Branch 1"))
year_2025.add_branch(Branch("Branch 2"))
database.append(year_2025)

# Print the database to verify the structure
for year in database:
    print(year)


# Q9):Using matplotlib plot the following functions on the domain: x = 0.5 to 100.0 in steps of 0.5.

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[40]:


x=np.arange(0.5,100.0,0.5)
y=x
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y,color='g')
plt.grid(True)
#plt.axis('equal')
plt.show()


# In[41]:


y=x**2
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[42]:


y=(x**3)/100
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[43]:


y=np.sin(x)
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('A sin curve')
plt.grid(True)
plt.show()


# In[44]:


y=np.sin(x)/x
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[45]:


y=np.log(x)
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[46]:


y=np.exp(x)
plt.plot(x,y,color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# Q10): Using numpy generate a matrix of size 20X5

# In[47]:


import numpy as np
import pandas as pd

A=np.random.uniform(1,2,size=(20,5))                   #generating random matrix
A
df=pd.DataFrame(A,columns=['a','b','c','d','e'])      #creating dataframe from matrix
df
C=np.argmax(df.std(axis=0))          # column with highest standard deviation
R=np.argmin(df.mean(axis=1))          #row with lowest mean
C,R


# Q11):Adding a new column to the dataframe called “f” which is the sum of the columns “a”, “b”, “c”, “d”, “e”. Create another column called “g”.

# In[48]:


df1=df.sum(axis=1)   #new column
df1.name='f'
df3=pd.concat([df,df1],axis=1)      #adding new column f to dataframe
g=[]
index=0
for value in df1:
    if value<8:
        g.append("LT8")
    else:
        g.append('GT8')
    index+=1
g
df4=pd.Series(g)
df4.name='g'
df5=pd.concat([df3,df4],axis=1)       #adding another column g
df5



details=df5.apply(lambda x: True if x['g']=='LT8' else False,axis=1)
num_rows=len(details[details==True].index)
print('number of rows in dataframe in which value is LT8 is:',num_rows)

#finding standard deviation

print(df5.loc[df5['g']=='LT8','f'].std())
print(df5.loc[df5['g']=='GT8','f'].std())


# Q.12):Piece of code to explain broadcasting in numpy.

# In[49]:


import numpy as np

# Define a 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Define a 1D array (vector)
vector = np.array([10, 20, 30])

# Adding the vector to the matrix
result = matrix + vector

print("Matrix:\n", matrix)
print("Vector:\n", vector)
print("Result:\n", result)


# Q13): Write a function to compute the argmin of a numpy array.

# In[50]:


import numpy as np

def compute_argmin(arr):
    """
    Compute the index of the minimum element in a NumPy array.

    Parameters:
    arr (np.ndarray): Input NumPy array.

    Returns:
    int: Index of the minimum element.
    """
    # Initialize min_index and min_value
    min_index = 0
    min_value = arr[0]
    
    # Iterate through the array to find the minimum value and its index
    for index in range(1, arr.size):
        if arr[index] < min_value:
            min_value = arr[index]
            min_index = index
    
    return min_index

# Example usage
array = np.array([5, 3, 8, 1, 9, 2])
print("Array:", array)

# Compute argmin using custom function
index_custom = compute_argmin(array)
print("Index of minimum element (custom function):", index_custom)

# Verify using np.argmin
index_np = np.argmin(array)
print("Index of minimum element (np.argmin):", index_np)

# Verify that both methods give the same result
assert index_custom == index_np, "The results from the custom function and np.argmin do not match!"


# In[ ]:




