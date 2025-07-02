## Catniff

Catniff is a small, experimental autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd). The name is a play on "catnip" and "differentiation".

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
const L = x.pow(2).add(x)

L.backward();
console.log(x.grad, L.grad);
```

All available APIs are in `./src/core.ts`.

## Todos

I'm mostly just learning and playing with this currently, so there are no concrete plans yet, but here are what I currently have in mind:

* A built-in Tensor maths lib.
* Support for Tensors in the autograd engine.
* GPU acceleration.
* Some general neural net APIs.

## Copyrights and License

Copyrights © 2025 Nguyen Phu Minh.

This project is licensed under the GPL 3.0 License.
