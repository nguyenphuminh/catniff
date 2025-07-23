# Catniff

Catniff is an experimental tensor ops library and autograd engine made to be Torch-like (its name is a play on "catnip" and "differentiation"). This project is heavily under development currently, so keep in mind that APIs can be completely unstable and backwards-incompatible.

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

## Documentation

Full documentation is available in [`./docs/documentation.md`](./docs/documentation.md).

All available APIs are in [`./src/core.ts`](./src/core.ts) if you want to dig deeper.

## Todos

I'm mostly just learning and playing with this currently, so there are no concrete plans yet, but here is what I currently have in mind:

* Fix whatever is the problem right now (there are a lot of problems right now lol).
* Add more tensor ops.
* Proper documentation.
* GPU acceleration.
* Some general neural net APIs.
* Refactor code.
* Proper tests.
* Option to load more backends.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
