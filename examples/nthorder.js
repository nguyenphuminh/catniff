const { Tensor } = require("../index");

Tensor.createGraph = true;

const x = new Tensor(2, { requiresGrad: true });
const L = x.pow(3).add(x);

// Result
console.log(`f = x^3 + x = ${L.val()}`)

// First-order grad
L.backward();
console.log(`f' = 3x^2 + 1 = ${x.grad.val()}`)

// Second-order grad
x.grad.backward();
console.log(`f'' = 6x = ${x.grad.val()}`);

// Third-order grad
x.grad.backward();
console.log(`f''' = ${x.grad.val()}`);
