### 1. What "linear" means

* **Linear regression** = model is **linear in its parameters (β’s)**.
* Features (x’s) can be transformed (squared, cubed, log, sqrt, etc.), but as long as β’s appear linearly, it’s still linear regression.

General form:

$$
y = β₀ + β₁f_1(x) + β₂f_2(x) + \dots + βₙf_n(x)
$$

---

### 2. Examples

* **Linear regression (straight line):**

  $$
  y = β₀ + β₁x
  $$

* **Polynomial regression (parabola, cubic, etc.):**

  $$
  y = β₀ + β₁x + β₂x^2 + β₃x^3
  $$

  Still linear regression (linear in β’s).

* **Log-transformed regression:**

  $$
  y = β₀ + β₁\log(x)
  $$

  Still linear in β’s.

* **Nonlinear regression (true nonlinear):**

  $$
  y = β₀ + e^{β₁x} \quad \text{or} \quad y = β₀ + (β₁)^2x
  $$

  Nonlinear in β’s → needs different optimization.

---

### 3. Polynomial Regression

* Technique: expand features into polynomial terms ($x, x^2, x^3, ...$) and run linear regression.
* Used to fit curves (parabola, cubic, etc.).
* Still solved with standard linear regression (ordinary least squares).

---

### 4. Why polynomial regression may not be optimal

* **Underfitting** if the real relation is more complex.
* **Overfitting** if the degree is too high (model fits noise).
* **Numerical issues** if features like $x^n$ get very large (fix with scaling/normalization).

Regularization (Ridge, Lasso) can help stabilize polynomial models.

---

### 5. Other regression types

* **Linear Regression** – straight line.
* **Polynomial Regression** – curved, via polynomial features.
* **Logistic Regression** – classification, not regression.
* **Ridge/Lasso/ElasticNet** – linear regression with regularization.
* **True Nonlinear Regression** – parameters inside nonlinear functions (needs iterative solvers).
* **Tree-based & Neural Networks** – flexible nonlinear models without explicit polynomial features.

---

### 6. Optimization and minima

* **Linear regression loss (MSE)** is convex → only one minimum (global minimum). Always solvable exactly.
* **Nonlinear regression loss** can have multiple valleys → optimizer may stop at a **local minimum** instead of the global one.

Analogy:

* Global minimum = lowest valley in the entire mountain range.
* Local minimum = a smaller dip nearby that isn’t the deepest.

---

👉 Key Takeaway:

* Linear regression = linear in parameters, not features.
* Polynomial regression = still linear regression, but features are powers of x.
* Nonlinear regression (true) = when parameters themselves appear nonlinearly.
* Polynomial regression works well for simple curves, but for complex patterns you may need regularization, trees, or neural networks.
