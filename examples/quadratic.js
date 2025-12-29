const { Tensor } = require("../index");

const x = new Tensor(2, { requiresGrad: true });
const L = x.pow(2).add(x);

L.backward();

console.log(x.grad.toString(), L.grad.toString());
