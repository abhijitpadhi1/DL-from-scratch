# I Built Backpropagation From Scratch (and It Broke Many Times)

### *What implementing a neural network for XOR taught me about gradients, bugs, and real learning*

Most deep learning tutorials start with something like this:

```python
model.fit(X, y)
```

And suddenly, magic happens.

This notebook was my attempt to remove the magic.

Instead of using TensorFlow or PyTorch, I decided to **build backpropagation from scratch**, using only NumPy — and test it on the smallest problem that still requires real learning: **XOR**.

What followed was a long debugging journey that taught me more than any high-level API ever did.

---

### 🔗 Code & Notebook

* [📁 **GitHub Repository**](https://colab.research.google.com/drive/1EIfvx7A_cDfOL1rMjDiAhRYRG2NlTMp_?usp=sharing)

* [▶️ **Google Colab Notebook (Run Instantly)**](https://github.com/abhijitpadhi1/DL-from-scratch/blob/main/MLP%20for%20XOR/Back_propagation_from_scratch.ipynb)

---

## Why XOR?

XOR looks trivial:

```
0 ⊕ 0 = 0  
0 ⊕ 1 = 1  
1 ⊕ 0 = 1  
1 ⊕ 1 = 0
```

But it has an important property:

> **XOR is not linearly separable.**

That single fact forces a neural network to:

* Use hidden layers
* Apply non-linear activations
* Backpropagate gradients correctly

If your network learns XOR, your implementation is *probably* correct.
If it doesn’t — something is broken.

---

### 🖼 Figure 1: XOR Is Not Linearly Separable

![XOR Is Not Linearly Separable](<Fig 1.png>)

> **Figure 1:** XOR is not linearly separable. No single straight line can separate the two classes, which makes hidden layers and non-linear activation functions necessary.

---

## What I Built (Briefly)

I implemented a **multi-layer perceptron from scratch**, including:

* A `Neuron` class

  * Stores weights, inputs, output, and gradients

* A `Layer` class

  * Handles bias and forward propagation

* A `MultiLayeredPerceptron` class

  * Manages architecture
  * Runs forward propagation
  * Implements backpropagation manually

No automatic differentiation.
No `.backward()` calls.
Every gradient was written by hand.

---

### 🖼 Figure 2: Network Architecture

![Network Architecture](<Fig 2.png>)

> **Figure 2:** Architecture of the multi-layer perceptron implemented from scratch.
> The input layer has 2 neurons, the first hidden layer has 2 neurons, the second hidden layer has 3 neurons, and the output layer has 1 neuron.

---

## Challenge #1: “It Runs” Does Not Mean “It Learns”

My first version ran perfectly.

No crashes.
No shape errors.
Weights were updating.

But the loss stayed stuck at **0.25** (or later **~0.69**) no matter how many iterations I ran.

That was my first big lesson:

> **A neural network can look correct and still be mathematically wrong.**

---

## Challenge #2: Broken Gradient Flow (The Silent Killer)

One of the hardest bugs had nothing to do with math.

It was architecture.

I built my network like this:

```python
model.add_hidden_layer(2)
model.add_hidden_layer(3)
model.add_output_layer(1)
```

But my layer-attachment logic only connected layers that already existed.

As a result:

* Hidden layers had **no reference to the output layer**
* Their `output_neurons` list was empty
* Their gradient (`delta`) was always zero

So during backpropagation:

```python
delta_sum = 0
self.delta = delta_sum * activation_derivative
```

The hidden layers **never learned anything**.

---

### 🖼 Figure 3: Gradient Flow in Backpropagation

![Gradient Flow in Backpropagation](<Fig 3.png>)

> **Figure 3:** Backpropagation requires uninterrupted gradient flow from the output layer to all hidden layers. A broken connection results in zero learning.

---

### What I learned

> **A neural network is a graph before it is math.
> If the graph is broken, gradients are zero.**

Once I fixed the layer connectivity, learning finally became possible.

---

## Challenge #3: Loss Function Mismatch

At one point, I used this for the output delta:

```python
self.delta = self.output - target
```

That’s correct — but **only for sigmoid + binary cross-entropy**.

However, I was *monitoring* training using mean squared error:

```python
error = (target - prediction) ** 2
```

So I was training with one objective and measuring another.

The result?

* Loss hovered around **log(2) ≈ 0.693**
* It looked like learning wasn’t happening

---

### What I learned

> **Loss, activation function, and delta must form one consistent chain rule.**

Once I aligned the loss with the gradient math, the loss curve finally made sense.

---

## The Turning Point: Watching the Loss Collapse

![Loss Collapse](<Plot 1.png>)

After fixing:

* Layer connectivity
* Bias updates
* Weight initialization
* Loss/gradient consistency

The loss curve finally did this:

* Stayed flat at first (random guessing)
* Suddenly dropped sharply
* Slowly converged close to zero

Seeing that curve fall was the moment I *knew* the network was truly learning.

---

## Visual Proof: Decision Boundary

![Decision Boundry](<Plot 2.png>)

The most satisfying part was visualizing the decision boundary.

Instead of a straight line (which XOR cannot use), the network learned **non-linear regions** that correctly separated the four XOR points.

This confirmed that:

* Hidden layers were transforming the input space
* The network wasn’t memorizing
* Backpropagation was working as intended

---

## What This Project Taught Me

This notebook taught me lessons that no framework abstraction ever did:

* Backpropagation is simple — but fragile
* Most learning bugs are **not math errors**, but:

  * Graph connectivity issues
  * Shape mismatches
  * Silent NumPy broadcasting
* Bias is not optional
* Visualization is a debugging tool, not just a presentation tool

Most importantly:

> **If you can debug your own backpropagation,
> you actually understand neural networks.**

---

## Final Thoughts

This wasn’t about performance or optimization.

It was about understanding *why* learning works — and *why* it fails.

If you’re learning deep learning and everything feels like magic, try this once:

* Implement XOR from scratch
* Break it
* Fix it
* Visualize it

You’ll never look at `.fit()` the same way again.

---

### 🔗 Explore the Code

* [📁 **GitHub Repository:**](https://colab.research.google.com/drive/1EIfvx7A_cDfOL1rMjDiAhRYRG2NlTMp_?usp=sharing)
* [▶️ **Google Colab Notebook:**](https://github.com/abhijitpadhi1/DL-from-scratch/blob/main/MLP%20for%20XOR/Back_propagation_from_scratch.ipynb)

---