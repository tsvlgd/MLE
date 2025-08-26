hey take this entire notes eberything is messed up non of the writtne formula anything else seems to be  unclear while pasting into notion, now what do i want is to rewritten notes with formula using text not in md format got it "---

## 3) Common loss functions (when to use)

- **Regression**
    - **MSE**: 1n∑(y−y^)2\frac{1}{n}\sum (y-\hat y)^2
        
        Smooth, differentiable, penalizes outliers strongly, convex in linear models.
        
    - **MAE**: 1n∑∣y−y^∣\frac{1}{n}\sum |y-\hat y|
        
        Robust to outliers, but gradient is undefined at 0 (use subgradients).
        
    - **Huber**: Quadratic near 0, linear in tails. Good compromise when outliers exist.
- **Classification**
    - **Binary cross-entropy**: −1n∑[ylog⁡p^+(1−y)log⁡(1−p^)]\frac{1}{n}\sum [y\log \hat p + (1-y)\log(1-\hat p)]
    - **Multiclass cross-entropy**: Softmax + NLL.
    - (SVM) **Hinge**: Margin-based; often used with linear SVMs.

### Formula

MSE=1n∑i=1n(yi−y^i)2\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2

- yiy_i = actual (observed) value
- y^i\hat y_i = predicted value from the model
- nn = number of data points

So it’s the **average of squared residuals (errors)**.

---

### 1. Smooth

- The function (y−y^)2(y - \hat y)^2 is a **parabola** (quadratic) in terms of y^\hat y.
- No sharp corners, no discontinuities → the error curve is smooth.
- That means optimization algorithms like **gradient descent** can move easily along it.

---

### 2. Differentiable

- You can take derivatives w.r.t model parameters.
- Gradient is:
    
    ∂∂y^  (y−y^)2=−2(y−y^)\frac{\partial}{\partial \hat y} \; (y - \hat y)^2 = -2 (y - \hat y)
    
- This derivative is what tells gradient descent in which direction to update parameters to reduce error.
- Because it’s differentiable everywhere, we can do calculus on it without issues (unlike MAE, which has a sharp corner at 0).

---

### 3. Penalizes outliers strongly

- Squaring the residuals makes large errors count much more than small ones.
- Example:
    - Residual = 2 → squared = 4
    - Residual = 10 → squared = 100
- So if you have an outlier, it can dominate the cost function.
- This is both good (forces model to fit major deviations) and bad (sensitive to noise/outliers).

---

### 4. Convex in linear models

- For **linear regression**, MSE as a function of parameters (w, b) forms a **convex bowl shape** (a paraboloid).
- Convex means: only **one global minimum**, no local traps.
- That guarantees gradient descent will converge to the optimal parameters (if learning rate is reasonable).
- In more complex models (like deep nets), MSE might not be globally convex, but in basic linear regression, it is.

---

### Quick numerical example

Say we have actual vs predicted values:

| y (actual) | ŷ (predicted) | residual | squared |
| --- | --- | --- | --- |
| 3 | 2 | 1 | 1 |
| 5 | 7 | -2 | 4 |
| 2 | 6 | -4 | 16 |

MSE=1+4+163=213=7\text{MSE} = \frac{1 + 4 + 16}{3} = \frac{21}{3} = 7

Notice: the big error (-4) contributed **16** to the cost, much larger than the small error (+1 → 1). That’s the outlier effect.

---

So in short:

MSE is chosen in linear regression because it’s **mathematically nice (smooth, differentiable, convex)** and also **statistically meaningful** (it’s the maximum likelihood estimator under Gaussian noise).

---

Do you want me to also contrast this with **MAE (Mean Absolute Error)** side by side, so you see *why* we use MSE more often in derivations?

---

## 4) Gradient Descent (GD)

You iteratively move parameters in the negative gradient direction:

- **Update rule:** θ←θ−α∇θJ(θ)\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
    - α = learning rate (too big → diverge; too small → slow).
- **Batch GD:** Uses whole dataset per step (stable, slower per step).
- **SGD:** Uses one example per step (noisy but can escape poor regions).
- **Mini-batch GD:** Small batches (most common in practice).
- **Tips:** Standardize features, monitor loss, use early stopping. For fundamentals, plain GD is enough; you don’t need Adam yet unless you’re doing deep nets.

---

## 5) Worked example (linear regression + MSE + one GD step)

**Data:**

x = [0, 1, 2],  y = [1, 3, 5]  (true relation is y = 2x + 1)

**Model:** y^=wx+b\hat y = w x + b

**Loss (MSE):** J(w,b)=13∑i=13(yi−(wxi+b))2J(w,b) = \frac{1}{3}\sum_{i=1}^3 (y_i - (w x_i + b))^2

**Gradients for MSE (linear regression):**

∂J∂w=−2n∑xi (yi−y^i),∂J∂b=−2n∑(yi−y^i)\frac{\partial J}{\partial w} = -\frac{2}{n}\sum x_i\,(y_i - \hat y_i), \quad
\frac{\partial J}{\partial b} = -\frac{2}{n}\sum (y_i - \hat y_i)

Start at w0=0, b0=0w_0 = 0,\, b_0 = 0. Let learning rate α=0.1\alpha = 0.1.

- Predictions at start: ŷ = [0, 0, 0]
    
    Residuals: r = y − ŷ = [1, 3, 5]
    
- MSE at start:
    
    J=13(12+32+52)=353≈11.6667J = \frac{1}{3}(1^2 + 3^2 + 5^2) = \frac{35}{3} \approx 11.6667
    
- Gradients:
    
    ∑r=1+3+5=9⇒∂J∂b=−23⋅9=−6\sum r = 1 + 3 + 5 = 9 \Rightarrow \frac{\partial J}{\partial b} = -\frac{2}{3}\cdot 9 = -6
    
    ∑xiri=0⋅1+1⋅3+2⋅5=13⇒∂J∂w=−23⋅13=−263\sum x_ir_i = 0\cdot1 + 1\cdot3 + 2\cdot5 = 13 \Rightarrow
    \frac{\partial J}{\partial w} = -\frac{2}{3}\cdot 13 = -\frac{26}{3}
    
- Update:
    
    w1=0−0.1(−263)=2630=1315≈0.8667w_1 = 0 - 0.1\left(-\frac{26}{3}\right) = \frac{26}{30} = \frac{13}{15} \approx 0.8667
    
    b1=0−0.1(−6)=0.6=35b_1 = 0 - 0.1(-6) = 0.6 = \frac{3}{5}
    
- New predictions:
    - x=0: ŷ = 3/5 → residual 1 − 3/5 = 2/5
    - x=1: ŷ = 13/15 + 3/5 = 22/15 → residual 3 − 22/15 = 23/15
    - x=2: ŷ = 26/15 + 3/5 = 35/15 = 7/3 → residual 5 − 7/3 = 8/3
- New MSE:
    
    J1=13[(25)2+(2315)2+(83)2]=13(425+529225+649)=13⋅2165225=433135≈3.2074J_1 = \frac{1}{3}\left[\left(\frac{2}{5}\right)^2 + \left(\frac{23}{15}\right)^2 + \left(\frac{8}{3}\right)^2\right]
        = \frac{1}{3}\left(\frac{4}{25} + \frac{529}{225} + \frac{64}{9}\right)
        = \frac{1}{3}\cdot\frac{2165}{225}
        = \frac{433}{135} \approx 3.2074
    

MSE dropped from ≈11.67 to ≈3.21 in one step—GD is working. Repeating steps will converge near w=2,b=1w=2, b=1.

---

## Minimal from-scratch code (optional)

```python
# Linear regression with GD on the tiny dataset
import math

x = [0.0, 1.0, 2.0]
y = [1.0, 3.0, 5.0]
w, b = 0.0, 0.0
alpha = 0.1

def mse(w, b):
    return sum((yi - (w*xi + b))**2 for xi, yi in zip(x, y)) / len(x)

def grads(w, b):
    n = len(x)
    dw = -(2.0/n) * sum(xi * (yi - (w*xi + b)) for xi, yi in zip(x, y))
    db = -(2.0/n) * sum((yi - (w*xi + b)) for xi, yi in zip(x, y))
    return dw, db

for step in range(20):
    dw, db = grads(w, b)
    w -= alpha * dw
    b -= alpha * db
    if step % 5 == 0:
        print(f"step={step:02d} w={w:.4f} b={b:.4f} mse={mse(w,b):.4f}")

```

---"  
