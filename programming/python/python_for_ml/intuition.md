
## 1. **Python execution flow (general)**

When you run Python code (script or notebook), these things happen:

1. **Source code**: Your `.py` file or notebook cell contains Python code.
2. **Lexing & Parsing**: The Python interpreter reads the text and converts it into **tokens**, then into a **parse tree**.
3. **Compilation to bytecode**: The parse tree is compiled into **Python bytecode** (`.pyc`).

   * Bytecode is **low-level instructions** understood by the Python Virtual Machine (PVM).
4. **Python Virtual Machine (PVM)**: This is the **runtime engine** inside the Python interpreter that executes bytecode line by line.

   * All memory management, variable allocation, function calls, and operations happen here.

> So when you do `a = 5 + 3`, Python first compiles it to bytecode like:

```
LOAD_CONST 5
LOAD_CONST 3
BINARY_ADD
STORE_NAME a
```

Then the PVM executes these instructions.

---

## 2. **Notebook (Jupyter) execution**

A **Jupyter notebook** is slightly different because it’s interactive:

1. **Frontend**: Your web browser shows the notebook interface (cells, markdown, outputs).
2. **Kernel**: Each notebook has a **kernel**, which is basically a **Python process** running in the background.

   * The kernel **runs your code**, keeps the variables in memory, executes functions, and stores outputs.
3. **Message passing**: The notebook sends your code to the kernel, the kernel executes it, and sends back results to display in the frontend.
4. **Stateful runtime**: Unlike a script that runs top-to-bottom and exits, the notebook kernel **retains the state**: variables, imports, and objects persist across cells until you restart the kernel.

---

## 3. **Kernel vs Python interpreter**

* **Python interpreter**: The standard program that runs `.py` scripts. Handles parsing, compilation, bytecode, and execution via the PVM.
* **Notebook kernel**: Essentially a Python interpreter running in **interactive mode**, capable of **receiving code, executing it, and returning outputs** while keeping state between executions.

---

## 4. **Putting it together**

| Environment      | Code input | Execution engine   | State persistence       |
| ---------------- | ---------- | ------------------ | ----------------------- |
| Python script    | `.py` file | Python interpreter | No (everything resets)  |
| Jupyter notebook | cell input | Notebook kernel    | Yes (variables persist) |

* Every notebook cell behaves like a **mini script** sent to the kernel.
* The kernel runs your Python code **via the same interpreter and bytecode compilation** as normal Python.
* That’s why you can mix `numpy`, `pandas`, or your custom functions — the kernel keeps everything loaded.

---

## 5. **IPython**

* The notebook kernel is often an **IPython kernel**, which is a richer interactive environment than vanilla Python.
* IPython adds features like:

  * Magic commands (`%timeit`, `%matplotlib inline`)
  * Rich object display (tables, plots, HTML)
  * Autocomplete and introspection
