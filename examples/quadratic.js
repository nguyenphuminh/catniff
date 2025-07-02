const { Node } = require("../index");

const x = new Node(2);
const L = x.pow(2).add(x)

L.backward();
console.log(x.grad, L.grad);
