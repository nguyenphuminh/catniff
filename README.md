# Catniff

Catniff is a small deep learning framework for Javacript, built to be Torch-like, but more direct on tensors and autograd usage like Tinygrad. This project is under development currently, so keep in mind that APIs can be unstable and backwards-incompatible. On a side-note, the name is a play on "catnip" and "differentiation".

## Setup

Install through npm:
```
npm install catniff
```

## Example

Here is a little demo of a quadratic function:
```js
const { Tensor } = require("catniff");

const x = new Tensor(2, { requiresGrad: true });
const L = x.pow(2).add(x); // x^2 + x

L.backward();

console.log(x.grad.val()); // 5
```

View all examples in [`./examples`](./examples).

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

And it can still do much more, check out the docs mentioned below for more information.

## Documentation

Full documentation is available in [`./docs/documentation.md`](./docs/documentation.md).

All available APIs are in [`./src/`](./src/) if you want to dig deeper.

## Todos

* Bug fixes.
* More tensor ops.
* GPU acceleration.
* Some general neural net APIs.
* More detailed documentation.
* Code refactoring.
* Proper tests.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
