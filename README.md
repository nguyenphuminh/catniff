# Catniff

Catniff is a small and experimental tensor library and autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd). The name is a play on "catnip" and "differentiation".

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

## Tensors

Tensors in Catniff are either numbers (scalars/0-D tensors) or multidimensional number arrays (n-D tensors).

There is a built-in `TensorMath` class to help with Tensor arithmetic, for example:
```js
const { TensorMath } = require("catniff");

const A = [ 1, 2, 3 ];
const B = 3
console.log(TensorMath.add(A, B));
```

All available APIs are in `./src/tensor.ts`.

## Autograd

To compute the gradient of our mathematical expression, we use the `Node` class to dynamically build our DAG:
```js
const { Node } = require("catniff");

const X = new Node([ 1, 2, 3 ]);
const L = x.pow(2).add(x); // X^2 + X

L.backward();
console.log(x.grad);
```

All available APIs are in `./src/autograd.ts`.

## Todos

I'm mostly just learning and playing with this currently, so there are no concrete plans yet, but here are what I currently have in mind:

* GPU acceleration.
* Some general neural net APIs.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
