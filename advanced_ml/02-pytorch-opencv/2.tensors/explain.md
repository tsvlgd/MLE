---

first principle

---

### **1. Why is NumPy better than manual computing and math?**

- **Speed:** NumPy uses optimized C and Fortran code under the hood, making operations much faster than manual Python loops.
- **Convenience:** It provides built-in functions for complex operations (e.g., matrix multiplication, statistical calculations) in just one line.
- **Memory Efficiency:** NumPy arrays store data more compactly than Python lists, reducing memory usage.
- **Broadcasting:** Automatically handles operations between arrays of different shapes, simplifying code.
- **Integration:** Works seamlessly with libraries like SciPy, Pandas, and TensorFlow.

---

### **2. Why are tensors better than NumPy arrays?**

- **GPU Acceleration:** Tensors (e.g., in PyTorch or TensorFlow) can run on GPUs, enabling massive speedups for large-scale computations.
- **Automatic Differentiation:** Tensors support gradient computation, which is essential for machine learning and deep learning.
- **Dynamic Computation Graphs:** Tensors allow for flexible, on-the-fly graph construction, unlike static NumPy arrays.
- **Scalability:** Tensors are designed for high-dimensional data (e.g., images, videos), while NumPy arrays are limited to lower-dimensional tasks.

---

### **3. What’s the first principle of tensors?**

The **first principle of tensors** is **generalization**:

- Tensors generalize scalars, vectors, and matrices to higher dimensions.
- A **0D tensor** is a scalar, a **1D tensor** is a vector, a **2D tensor** is a matrix, and **nD tensors** extend this to any number of dimensions.
- They preserve mathematical properties (e.g., linear transformations) regardless of dimensionality.

---

### Data types

---

![image.png](attachment:4a3682a9-d750-49d8-8190-e66fd9ec8277:image.png)

### dot or matmul

---

| Feature | torch.dot | torch.matmul (@) |
| --- | --- | --- |
| **Input** | 1D tensors (vectors) | nD tensors (matrices, etc.) |
| **Output** | Scalar | Tensor (e.g., matrix) |
| **Use Case** | Vector similarity, norms | Neural networks, transforms |

---

### **1. `torch.nn.functional.softmax`**

- **Purpose**: Converts a vector of raw scores (logits) into probabilities that sum to 1.
- **Formula**:
softmax(xi)=∑jexjexi
    
    softmax(xi)=exi∑jexj\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
    
- **Use Case**: Multi-class classification (e.g., output layer of a neural network).
- **Example**:
    
    ```python
    import torch.nn.functional as F
    logits = torch.tensor([1.0, 2.0, 3.0])
    probabilities = F.softmax(logits, dim=0)
    # Output: tensor([0.0900, 0.2447, 0.6652])  # Sums to 1
    
    ```
    

---

### **2. `torch.nn.functional.relu` (Rectified Linear Unit)**

- **Purpose**: Introduces non-linearity by zeroing out negative values.
- **Formula**:
ReLU(x)=max(0,x)
    
    ReLU(x)=max⁡(0,x)\text{ReLU}(x) = \max(0, x)
    
- **Use Case**: Hidden layers in neural networks (fast, avoids vanishing gradients).
- **Example**:
    
    ```python
    x = torch.tensor([-1.0, 0.0, 2.0])
    output = F.relu(x)
    # Output: tensor([0., 0., 2.])
    
    ```
    

---

### **3. `torch.nn.functional.sigmoid`**

- **Purpose**: Maps any input to a value between 0 and 1.
- **Formula**:
sigmoid(x)=1+e−x1
    
    sigmoid(x)=11+e−x\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
    
- **Use Case**: Binary classification (e.g., output layer for yes/no predictions).
- **Example**:
    
    ```python
    x = torch.tensor([-1.0, 0.0, 1.0])
    output = torch.sigmoid(x)
    # Output: tensor([0.2689, 0.5000, 0.7311])
    
    ```
    

---

### **Key Differences**

| Function | Output Range | Use Case | Formula |
| --- | --- | --- | --- |
| `softmax` | (0, 1) | Multi-class classification | exi/∑jexje^{x_i} / \sum_j e^{x_j}exi/∑jexj |
| `ReLU` | [0, ∞) | Hidden layers | max⁡(0,x)\max(0, x)max(0,x) |
| `sigmoid` | (0, 1) | Binary classification | 1/(1+e−x)1 / (1 + e^{-x})1/(1+e−x) |

---

## **1. Manual Seed (`torch.manual_seed`)**

### **Purpose**

- Ensures **reproducibility** of random operations (e.g., tensor initialization, shuffling).
- Useful for debugging or comparing experiments.

### **How to Use**

```python
import torch

# Set a manual seed (e.g., 42)
torch.manual_seed(42)

# Example: Random tensor will now be reproducible
random_tensor = torch.rand(3, 3)
print(random_tensor)

```

- **Note**: For full reproducibility (especially with CUDA), also use:
    
    ```python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ```
    

---

## **2. In-Place Operations**

### **Purpose**

- Modifies a tensor **without allocating new memory**, saving resources.
- Recognizable by the `_` suffix (e.g., `.add_()`, `.relu_()`).

### **Common In-Place Functions**

| Function | In-Place Equivalent |
| --- | --- |
| `tensor.add()` | `tensor.add_()` |
| `tensor.relu()` | `tensor.relu_()` |
| `tensor.zero_()` | Sets all elements to 0 |

### **Example**

```python
x = torch.tensor([1, 2, 3])
x.add_(1)# In-place: x becomes [2, 3, 4]

```

### **Caveats**

- **Autograd Issues**: In-place ops can break gradient tracking in PyTorch’s autograd system. Use with caution in training loops.
- **Error Example**:
    
    ```python
    x = torch.tensor([1.0], requires_grad=True)
    x.add_(1)# Raises: "a leaf Variable that requires grad is being used in an in-place operation."
    
    ```
    

---

### **When to Use**

- **Manual Seed**: Always use for reproducibility (e.g., research, debugging).
- **In-Place Ops**: Use for memory efficiency, but avoid in autograd contexts.

---

## **1. Shallow Copy (`=`, `.copy_()`)**

- **Behavior**: Creates a **new reference** to the same underlying data.
- **Memory**: Shares memory with the original tensor.
- **Gradients**: If the original tensor has `requires_grad=True`, changes to the shallow copy affect the original and vice versa.
- **Example**:
    
    ```python
    import torch
    x = torch.tensor([1, 2, 3], requires_grad=True)
    y = x# Shallow copy (new reference)
    y[0] = 100# Modifies x as well
    print(x)# Output: tensor([100,   2,   3], requires_grad=True)
    
    ```
    

---

## **2. Deep Copy (`.clone()`)**

- **Behavior**: Creates a **new tensor** with a **copy of the data**.
- **Memory**: Does **not** share memory with the original tensor.
- **Gradients**:
    - If the original tensor has `requires_grad=True`, the clone will also have `requires_grad=True` **but** will not share the gradient history.
    - Changes to the clone do **not** affect the original tensor.
- **Example**:
    
    ```python
    x = torch.tensor([1, 2, 3], requires_grad=True)
    y = x.clone()# Deep copy
    y[0] = 100# Does not modify x
    print(x)# Output: tensor([1, 2, 3], requires_grad=True)
    print(y)# Output: tensor([100,   2,   3], requires_grad=True)
    
    ```
    

---

## **3. `.detach()` vs `.clone()`**

- **`.detach()`**: Creates a **new tensor** that shares data but **does not require gradients**.
    
    ```python
    y = x.detach()# Shares data, no grad
    
    ```
    
- **`.clone()`**: Creates a **new tensor** with a **copy of the data** and preserves `requires_grad` if specified.
    
    ```python
    y = x.clone()# New data, preserves grad if requires_grad=True
    
    ```
    

---

## **Key Differences**

| Method | Memory Sharing | Gradient Tracking | Use Case |
| --- | --- | --- | --- |
| Shallow Copy (`=`) | Yes | Shared | Avoid (can lead to unintended side effects) |
| `.clone()` | No | Preserved | Safe copy with grad tracking |
| `.detach()` | Yes | Disabled | Copy without grad tracking |

## **When to Use Which?**

- Use **`.clone()`** when you need a **fully independent copy** of a tensor (data + grad).
- Use **`.detach()`** when you want to **remove gradient tracking** but keep the data.
- Avoid **shallow copies** (`=`) for tensors with `requires_grad=True` to prevent unintended side effects.

---

### **Gradient Tracking in PyTorch**

Gradient tracking is a core feature of PyTorch’s **automatic differentiation (autograd)** system. It allows PyTorch to **automatically compute gradients** (derivatives) of tensors with respect to some loss function, which is essential for training neural networks using **backpropagation**.

---

## **How It Works**

1. **`requires_grad=True`**:
    - When you create a tensor with `requires_grad=True`, PyTorch **tracks all operations** performed on it.
    - This builds a **computational graph** that records how the tensor was computed.
2. **Forward Pass**:
    - During the forward pass, PyTorch records the operations (e.g., additions, multiplications) applied to the tensor.
3. **Backward Pass**:
    - When you call `.backward()` on a tensor (usually the loss), PyTorch **traverses the computational graph backward** and computes the gradients for all tensors with `requires_grad=True`.

---

## **Example**

```python
import torch

# Create a tensor with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform an operation (forward pass)
y = x * 2
z = y.sum()

# Compute gradients (backward pass)
z.backward()

# Access the gradients
print(x.grad)# Output: tensor([2., 2., 2.])

```

- Here, `x.grad` contains the gradient of `z` with respect to `x`, which is `2` for each element because ∂xi∂z=2.
    
    ∂z∂xi=2\frac{\partial z}{\partial x_i} = 2
    

---

## **Key Concepts**

1. **`requires_grad`**:
    - If `True`, PyTorch tracks operations on the tensor.
    - If `False`, the tensor is treated as a constant (no gradient tracking).
2. **`grad` Attribute**:
    - After calling `.backward()`, the gradients are stored in the `.grad` attribute of the tensor.
3. **`detach()`**:
    - Creates a new tensor that **shares data** but **does not require gradients**.
    - Useful when you want to stop gradient tracking for intermediate computations.
4. **`with torch.no_grad():`**:
    - Temporarily disables gradient tracking within a block of code.
    - Useful for inference or updating model weights manually.

---

## **Why It Matters**

- **Training Neural Networks**: Gradient tracking enables backpropagation, allowing the model to learn from data.
- **Dynamic Computation Graphs**: PyTorch’s autograd system dynamically builds the graph during the forward pass, making it flexible for complex models.
- **Memory Efficiency**: You can disable gradient tracking for tensors that don’t need it (e.g., inference), saving memory and computation.

---

### **Batching in Deep Learning**

**Batching** is the process of dividing a dataset into smaller subsets (called **batches**) to train a model efficiently. Instead of feeding the entire dataset at once, the model processes one batch at a time. This approach is crucial for **memory efficiency**, **faster training**, and **better generalization**.

---

## **Why Use Batching?**

1. **Memory Efficiency**: GPUs/CPUs have limited memory. Batching allows you to train on large datasets without running out of memory.
2. **Faster Training**: Parallel processing (e.g., on GPUs) is optimized for batches.
3. **Smoother Gradient Updates**: Updates weights using the average gradient of a batch, reducing noise compared to single-sample updates (stochastic gradient descent).

---

## **How Batching Works (Example: Image Data)**

Let’s say you have a dataset of **10,000 images**, each of size **3x224x224** (3 channels, 224x224 pixels).

### **Step 1: Define Batch Size**

- Choose a **batch size** (e.g., 32, 64, or 128). This is the number of images processed in one forward/backward pass.
- Example: Batch size = **32**.

### **Step 2: Organize Data into Batches**

- The dataset is split into batches of 32 images each:
    - Batch 1: Images 1–32
    - Batch 2: Images 33–64
    - ...
    - Batch 313: Images 9985–10000 (last batch may be smaller).

### **Step 3: Shape of a Batch**

- For a batch of 32 images, the tensor shape is:
**`(batch_size, channels, height, width) = (32, 3, 224, 224)`**.

---

## **How Batches Are Accepted in PyTorch**

In PyTorch, you typically use:

1. **`DataLoader`**: Automatically splits the dataset into batches and shuffles them.
2. **`Dataset`**: Defines how to load individual samples (e.g., images and labels).

### **Example Code**

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images# List of images (each: 3x224x224)
        self.labels = labels# List of labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create dataset and DataLoader
dataset = ImageDataset(images, labels)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches
for batch_images, batch_labels in dataloader:
    print(batch_images.shape)# Output: torch.Size([32, 3, 224, 224])# Train your model on this batch

```

---

## **Key Points**

- **Batch Size**: A hyperparameter (e.g., 32, 64, 128). Larger batches speed up training but require more memory.
- **Shuffling**: Batches are shuffled to avoid order bias (e.g., `shuffle=True` in `DataLoader`).
- **Last Batch**: May be smaller if the dataset isn’t divisible by the batch size.

---