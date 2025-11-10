# Catniff 😺🌿

Catniff is a small deep learning framework for Javacript, built to be Torch-like, but more direct on tensors and autograd usage like Tinygrad. This project is under development currently, so keep in mind that APIs can be unstable and backwards-incompatible. On a side-note, the name is a play on "catnip" and "differentiation".

## Setup

Install through npm:
```
npm install catniff
```

## Tensors

Tensors in Catniff can be created by passing in a number or an nD array, and there are built-in methods that can be used to perform tensor arithmetic:
```js
const { Tensor } = require("catniff");

// Tensor init
const A = new Tensor([ 1, 2, 3 ]);
const B = new Tensor(3);

// Tensor addition (.val() returns the raw value rather than the tensor object)
console.log(A.add(B).val());
```

## Autograd

To compute the gradient wrt multiple variables of our mathematical expression, we can simply set `requiresGrad` to `true`:
```js
const { Tensor } = require("catniff");

const X = new Tensor(
    [
        [ 0.5, -1.0 ],
        [ 2.0,  0.0 ]
    ],
    { requiresGrad: true }
);

const Y = new Tensor(
    [
        [ 1.0, -2.0 ],
        [ 0.5,  1.5 ]
    ],
    { requiresGrad: true }
);

const D = X.sub(Y);
const E = D.exp();
const F = E.add(1);
const G = F.log();

G.backward();

// X.grad and Y.grad are tensor objects themselves, so we call .val() here to see their raw values
console.log(X.grad.val(), Y.grad.val());
```

## Optimizer

Catniff comes bundled with optimizers as well:
```js
const { Tensor, Optim } = require("catniff");

// Define some parameter
const w = new Tensor([1.0], { requiresGrad: true });
// Define a fake loss function: L = (w - 3)^2
const loss = w.sub(3).pow(2);
// Calculate gradient
loss.backward();
// Use Adam optimizer
const optim = new Optim.Adam([w]);
// Optimization step
optim.step();

console.log("Updated weight:", w.data);  // Should move toward 3.0
```

## Neural networks & Deep learning

There are built-in neural network constructs in Catniff as well, from simple prebuilt nn layers:
```js
const { Tensor, nn } = require("catniff");

// Linear layer with input size of 20 and output size of 10
const linear = nn.Linear(20, 10);
// RNN cell with input size of 32 and hidden size of 64
const rnnCell = nn.RNNCell(32, 64);
// Same thing but using GRU
const gruCell = nn.GRUCell(32, 64);
// Same thing but using LSTM
const lstmCell = nn.LSTMCell(32, 64);

// Forward passes
const a = Tensor.randn([20]);
const b = Tensor.randn([32]);
const c = Tensor.randn([64]);

linear.forward(a);
rnnCell.forward(b, c);
gruCell.forward(b, c);
lstmCell.forward(b, c, c);
```

to more advanced constructs like normalization, embedding, and attention:
```js
// 1. Embedding: tokens -> vectors
const embedding = new nn.Embedding(100, 64);
const tokens = new Tensor([[1, 5, 23], [8, 2, 15]]);
const embedded = embedding.forward(tokens);

// 2. Self-Attention
const attention = new nn.MultiheadAttention(64, 8, 0.1);
const [output, weights] = attention.forward(embedded, embedded, embedded);

// 3. Layer Normalization
const layerNorm = new nn.LayerNorm(64);
const normalized = layerNorm.forward(output);

console.log(normalized.val());
```

And it can still do much more, check out the docs and examples below for more information.

## Documentation

Full documentation is available in [`./docs/documentation.md`](./docs/documentation.md).

All available APIs are in [`./src/`](./src/) if you want to dig deeper.

## Examples

* [Shakespeare-style text generator](https://github.com/nguyenphuminh/shakespeare-lm).
* [Simple neural net for XOR calculation](./examples/xornet.js).
* [N-th order derivative calculation](./examples/nthorder.js).
* [Tensors](./examples/tensors.js).
* [Optimizer](./examples/optim.js).
* [Simple quadratic equation](./examples/quadratic.js).

## Todos

* More general tensor ops.
* More general neural net APIs.
* GPU acceleration.
* Comprehensive caching.
* Bug fixes.
* More detailed documentation.
* Code refactoring.
* Proper tests.

## Cite Catniff

```bibtex
@misc{catniff,
  author = {Phu Minh Nguyen},
  title = {Catniff: Torch-like deep learning framework for Javascript},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nguyenphuminh/catniff}
}
```

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the Apache 2.0 license.
