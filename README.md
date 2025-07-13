# Catniff

Catniff is an experimental tensor ops library and autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd), and its name is a play on "catnip" and "differentiation". The project is heavily in-dev currently, so keep in mind that APIs can be unstable and backwards-incompatible.

## Setup

Install through npm:
```
npm install catniff
```

## Example

Here is a little demo of a quadratic function:
```js
const { Node } = require("catniff");

const x = new Node(2);
const L = x.pow(2).add(x); // x^2 + x

L.backward();
console.log(x.grad); // 5
```

View all examples in [`./examples`](./examples).

## Tensors

Tensors in Catniff are either numbers (scalars/0-D tensors) or multidimensional number arrays (n-D tensors).

There is a built-in `TensorMath` class to help with tensor arithmetic, for example:
```js
const { TensorMath } = require("catniff");

const A = [ 1, 2, 3 ];
const B = 3;
console.log(TensorMath.add(A, B));
```

If you want to be concise, you can use `TM` or `TMath`:
```js
const { TM, TMath } = require("catniff");

const A = [ 1, 2, 3 ];
const B = 3;
console.log(TM.add(A, B));
console.log(TMath.add(A, B));
```

All available APIs are in `./src/tensor.ts`.

## Autograd

To compute the gradient wrt multiple variables of our mathematical expression, we use the `Node` class to dynamically build our DAG:
```js
const { Node } = require("catniff");

const X = new Node([
    [ 0.5, -1.0 ],
    [ 2.0,  0.0 ]
]);

const Y = new Node([
    [ 1.0, -2.0 ],
    [ 0.5,  1.5 ]
]);

const D = X.sub(Y);
const E = D.exp();
const F = E.add(1);
const G = F.log();

G.backward();

console.log(X.grad, Y.grad);
```

All available APIs are in `./src/autograd.ts`.

## Todos

I'm mostly just learning and playing with this currently, so there are no concrete plans yet, but here is what I currently have in mind:

* Fix whatever is the problem right now (there are a lot of problems right now lol).
* Add more tensor ops.
* Proper documentation.
* GPU acceleration.
* Some general neural net APIs.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
