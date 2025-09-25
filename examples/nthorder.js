const { Tensor } = require("../index");

Tensor.createGraph = true;

const x = new Tensor(2, { requiresGrad: true });
const L = x.pow(3).add(x);

L.backward();

console.log(`f = x^3 + x = ${L.val()}`)
console.log(`f' = 3x^2 + 1 = ${x.grad.val()}`)

const grad1 = x.grad.clone();

grad1.backward();

console.log(`f'' = 6x = ${x.grad.val()}`);

const grad2 = x.grad.clone();

grad2.backward();

console.log(`f''' = ${x.grad.val()}`);
